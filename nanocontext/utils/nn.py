import torch
import torch.nn.functional as F


def rms_norm(x):
    return F.rms_norm(x, (x.size(-1),))


def rotary_emb_attn(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:d + d]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], dim=3)
