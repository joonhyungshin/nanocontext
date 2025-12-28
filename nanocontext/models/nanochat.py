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

    def forward(self, x, cos_sin, kv_cache):
        x = x + self.attn(rms_norm(x), cos_sin, kv_cache)
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
        cos, sin = self._precompute_rotary_embd(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _precompute_rotary_embd(self, seq_len, head_dim, base=100000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freq = torch.outer(t, inv_freq)
        cos, sin = freq.cos(), freq.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def forward(self, x, targets=None, kv_cache=None, loss_reduction="mean"):
        B, T = x.size()

        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

        x = self.transformer.wte(x)
        x = rms_norm(x)
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache)
        x = rms_norm(x)

        softcap = 15
        logits = self.lm_head(x)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
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
            with autocast():
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

    def init_weights(self):
        self.apply(self._init_weights)
        torch.nn.init.zeros_(self.lm_head.weight)
        for block in self.transformer.h:
            torch.nn.init.zeros_(block.mlp.proj.weight)
            torch.nn.init.zeros_(block.attn.proj.weight)
        head_dim = self.config.n_embd // self.config.n_heads
        cos, sin = self._precompute_rotary_embd(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)
