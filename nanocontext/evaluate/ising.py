import math

import numpy as np
import torch
import torch.distributed as dist

from nanocontext.data.broadcast_tree import Engine
from nanocontext.utils import ddp_world_size


@torch.inference_mode()
def sample_magnets(engine: Engine, prompt, num_samples, max_tokens,
                   batch_samples=None, max_summary_tokens=64):
    batch_samples = batch_samples or num_samples
    magnet = torch.zeros(num_samples, device=engine.device)
    tokenizer = engine.tokenizer
    pos_token = tokenizer.tokenize_value(1)
    neg_token = tokenizer.tokenize_value(-1)
    for i in range(0, num_samples, batch_samples):
        actual_batch_samples = min(num_samples - i, batch_samples)
        for token_tensor in engine.generate_tree_tokens_tensor_stream(prompt,
                                                                      max_tokens=max_tokens,
                                                                      allow_many=True,
                                                                      num_samples=actual_batch_samples,
                                                                      max_context_tokens=max_summary_tokens):
            spin_tensor = (token_tensor == pos_token).int() - (token_tensor == neg_token).int()
            magnet[i:i + actual_batch_samples] += spin_tensor
    return magnet


def gather_magnets(engine: Engine, prompt, total_samples, max_tokens,
                   batch_samples=None):
    world_size = ddp_world_size()
    num_samples = (total_samples + world_size - 1) // world_size
    total_samples = num_samples * world_size
    magnet = sample_magnets(engine, prompt, num_samples, max_tokens, batch_samples=batch_samples)
    if world_size > 1:
        magnets = torch.empty(total_samples, dtype=magnet.dtype, device=magnet.device)
        dist.all_gather_into_tensor(magnets, magnet)
        return magnets
    else:
        return magnet


def compute_moments(magnet):
    n = len(magnet)
    total_magnet = np.sum(magnet)
    mean_magnet = total_magnet / n
    magnet_var = np.sum((magnet - mean_magnet) ** 2)
    magnet_fourth = np.sum((magnet - mean_magnet) ** 4)
    biased_var = magnet_var / n
    biased_fourth = magnet_fourth / n
    unbiased_var = magnet_var / (n - 1)
    sample_kurtosis = biased_fourth / biased_var ** 2 - 3
    fisher_kurtosis = (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * sample_kurtosis + 6)
    return unbiased_var, fisher_kurtosis


def evaluate_moments(engine: Engine, prompt, total_samples, max_tokens,
                     batch_samples=None, actual_tokens_hint=None):
    """Computes sample variance and excess kurtosis."""
    magnet_tensor = gather_magnets(engine, prompt, total_samples, max_tokens, batch_samples=batch_samples)
    magnet = magnet_tensor.detach().cpu().numpy()
    normalized_magnet = magnet / math.sqrt(actual_tokens_hint or max_tokens)
    return compute_moments(normalized_magnet)
