from contextlib import nullcontext

import torch

from .dist import device_to_use


def d_divide(n, d):
    cnt = 0
    while n % d == 0:
        cnt += 1
        n //= d
    return n, cnt


def d_order(n, d):
    return d_divide(n, d)[1]


def autocast(device=None):
    device = device or device_to_use()
    device_type = device.type
    if device_type == "cuda":
        return torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
    return nullcontext()


def synchronize(device=None):
    device = device or device_to_use()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
