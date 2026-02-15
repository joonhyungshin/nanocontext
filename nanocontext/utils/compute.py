from contextlib import nullcontext

import numpy as np
import torch

from .dist import device_to_use


def d_divide(n, d):
    assert n > 0
    cnt = 0
    while n % d == 0:
        cnt += 1
        n //= d
    return n, cnt


def d_order(n, d):
    return d_divide(n, d)[1]


def compute_moments(x):
    n = len(x)
    x_mean = np.mean(x)
    x_var = np.sum((x - x_mean) ** 2)
    x_fourth = np.sum((x - x_mean) ** 4)
    biased_var = x_var / n
    biased_fourth = x_fourth / n
    sample_kurtosis = biased_fourth / biased_var ** 2 - 3
    fisher_kurtosis = (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * sample_kurtosis + 6)
    unbiased_var = x_var / (n - 1)
    return unbiased_var, fisher_kurtosis


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


def plot_histogram(wandb_data, **plot_kwargs):
    import matplotlib.pyplot as plt

    data = wandb_data["magnets"]
    values, bins = data["values"], data["bins"]
    widths = np.diff(bins)
    plt.bar(bins[:-1], values, width=widths, align="edge", **plot_kwargs)
    plt.show()
