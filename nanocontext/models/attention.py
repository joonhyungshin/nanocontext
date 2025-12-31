import torch
import torch.nn as nn
import torch.nn.functional as F

from nanocontext.utils import rotary_emb_attn, rms_norm


class KVCache:
    def __init__(self, batch_size, n_heads, seq_len, head_dim, n_layers):
        self.kv_shape = (n_layers, 2, batch_size, n_heads, seq_len, head_dim)
        self.kv_cache = None
        self.pos = 0

    def get_pos(self):
        return self.pos

    def insert(self, layer_idx, k, v):
        if self.kv_cache is None:
            self.kv_cache = torch.empty(self.kv_shape, dtype=k.dtype, device=k.device)
        B, H, T_diff, D = k.size()
        t0, t1 = self.pos, self.pos + T_diff
        if t1 > self.kv_cache.size(4):
            t_new = t1 + 1024
            t_new = (t_new + 1023) & ~1023
            shape_new = list(self.kv_cache.shape)
            shape_new[4] = t_new - self.kv_cache.size(4)
            cache_new = torch.empty(shape_new, dtype=k.dtype, device=k.device)
            self.kv_cache = torch.cat([self.kv_cache, cache_new], dim=4).contiguous()
            self.kv_shape = self.kv_cache.shape
        self.kv_cache[layer_idx, 0, :, :, t0:t1, :] = k
        self.kv_cache[layer_idx, 1, :, :, t0:t1, :] = v
        k_view = self.kv_cache[layer_idx, 0, :, :, :t1, :]
        v_view = self.kv_cache[layer_idx, 1, :, :, :t1, :]
        if layer_idx == self.kv_cache.size(0) - 1:
            self.pos = t1
        return k_view, v_view

    def copy_from(self, other):
        dtype, device = other.kv_cache.dtype, other.kv_cache.device
        self.kv_cache = torch.empty(self.kv_shape, dtype=dtype, device=device)
        self.kv_cache[:, :, :, :, :other.pos, :] = other.kv_cache
        self.pos = other.pos


class CausalSelfAttention(nn.Module):
    def __init__(self, layer_idx, n_heads, n_kv_heads, n_embd):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_embd = n_embd
        self.head_dim = self.n_embd // self.n_heads
        self.w_q = nn.Linear(n_embd, n_heads * self.head_dim, bias=False)
        self.w_k = nn.Linear(n_embd, n_kv_heads * self.head_dim, bias=False)
        self.w_v = nn.Linear(n_embd, n_kv_heads * self.head_dim, bias=False)
        self.proj = nn.Linear(n_heads * self.head_dim, n_embd, bias=False)

    def forward(self, x, cos_sin, kv_cache):
        B, T, C = x.size()

        q = self.w_q(x).view(B, T, self.n_heads, self.head_dim)
        k = self.w_k(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.w_v(x).view(B, T, self.n_kv_heads, self.head_dim)

        cos, sin = cos_sin
        q, k = rotary_emb_attn(q, cos, sin), rotary_emb_attn(k, cos, sin)
        q, k = rms_norm(q), rms_norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if kv_cache is not None:
            k, v = kv_cache.insert(self.layer_idx, k, v)
        T_q = q.size(2)
        T_k = k.size(2)

        enable_gqa = self.n_heads != self.n_kv_heads
        if kv_cache is None or T_q == T_k:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
        elif T_q == 1:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)
        else:
            attn_mask = torch.zeros((T_q, T_k), dtype=torch.bool, device=q.device)
            prefix_len = T_k - T_q
            attn_mask[:, :prefix_len] = True
            attn_mask[:, prefix_len:] = torch.tril(torch.ones((T_q, T_q), dtype=torch.bool, device=q.device))
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)

        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.proj(y)
        return y
