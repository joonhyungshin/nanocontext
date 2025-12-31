"""
Adapted from nanochat.
"""
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn
from nanocontext.utils import rms_norm, autocast

from .attention import CausalSelfAttention


@dataclass
class NanochatConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layers: int = 12
    n_heads: int = 6
    n_kv_heads: int = 6
    n_embd: int = 768
    rotary_embd_base: int = 10000


class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.proj = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x).square()
        x = self.proj(x)
        return x


class Block(nn.Module):
    def __init__(self, layer_idx, n_heads, n_kv_heads, n_embd):
        super().__init__()
        self.attn = CausalSelfAttention(layer_idx, n_heads, n_kv_heads, n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x, rotation, kv_cache):
        x = x + self.attn(rms_norm(x), rotation, kv_cache)
        x = x + self.mlp(rms_norm(x))
        return x


class Nanochat(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(layer_idx, config.n_heads, config.n_kv_heads, config.n_embd)
                                for layer_idx in range(config.n_layers)])
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_heads
        rotary_emb_shape = (1, self.rotary_seq_len, 1, head_dim // 2)
        self.register_buffer("rotary_embd_cos",
                             torch.empty(rotary_emb_shape, dtype=torch.bfloat16), persistent=False)
        self.register_buffer("rotary_embd_sin",
                             torch.empty(rotary_emb_shape, dtype=torch.bfloat16), persistent=False)
        self._precompute_rotary_embd()

    def _precompute_rotary_embd(self):
        device = self.transformer.wte.weight.device
        head_dim = self.config.n_embd // self.config.n_heads
        channel_range = torch.arange(0, head_dim - 1, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (self.config.rotary_embd_base ** (channel_range / head_dim))
        t = torch.arange(self.rotary_seq_len, dtype=torch.float32, device=device)
        freq = torch.outer(t, inv_freq)
        cos, sin = freq.cos(), freq.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        self.rotary_embd_cos, self.rotary_embd_sin = cos[None, :, None, :], sin[None, :, None, :]

    def preprocess(self):
        self._precompute_rotary_embd()
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)

    def forward(self, x, kv_cache=None):
        B, T = x.size()

        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        rotation = self.rotary_embd_cos[:, T0:T0+T], self.rotary_embd_sin[:, T0:T0+T]

        x = self.transformer.wte(x)
        x = rms_norm(x)
        for block in self.transformer.h:
            x = block(x, rotation, kv_cache)
        x = rms_norm(x)

        softcap = 15
        logits = self.lm_head(x)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)
        return logits

    @property
    def device(self):
        return self.transformer.wte.weight.device

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=None):
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=self.device)
            if seed is not None:
                rng.manual_seed(seed)
        x = torch.tensor([tokens], dtype=torch.long, device=self.device)
        for _ in range(max_tokens):
            logits = self.forward(x)
            logits = logits[:, -1, :]
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_x = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_x = torch.argmax(logits, dim=-1, keepdim=True)
            x = torch.cat([x, next_x], dim=-1)
            token = next_x.item()
            yield token
