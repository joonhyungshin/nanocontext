from contextlib import nullcontext

import torch

from .dist import device_to_use


def d_order(n, d):
    cnt = 0
    while n % d == 0:
        cnt += 1
        n //= d
    return cnt


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
