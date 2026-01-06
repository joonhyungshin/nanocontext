import math

import torch
import torch.distributed as dist

from .sample import NanochatSampler
from .utils import ddp_world_size


@torch.inference_mode()
def sample_magnets(model, tokenizer, num_samples, max_tokens,
                   batch_samples=None, seed=None):
    batch_samples = batch_samples or num_samples
    magnet = torch.empty(num_samples, device=model.device)
    sampler = NanochatSampler(model, seed=seed)
    for i in range(0, num_samples, batch_samples):
        actual_batch_samples = min(num_samples - i, batch_samples)
        tokens = sampler.generate_batch_tensor([0], num_samples=actual_batch_samples, max_tokens=max_tokens)
        magnet[i:i + actual_batch_samples] = (torch.sum(tokens == tokenizer.pos_token, dim=1) -
                                              torch.sum(tokens == tokenizer.neg_token, dim=1))
    return magnet


def gather_magnets(model, tokenizer, total_samples, max_tokens,
                   batch_samples=None, seed=None):
    world_size = ddp_world_size()
    num_samples = (total_samples + world_size - 1) // world_size
    total_samples = num_samples * world_size
    magnet = sample_magnets(model, tokenizer, num_samples, max_tokens, batch_samples=batch_samples, seed=seed)
    if world_size > 1:
        magnets = torch.empty(total_samples, dtype=magnet.dtype, device=magnet.device)
        dist.all_gather_into_tensor(magnets, magnet)
        return magnets
    else:
        return magnet


def evaluate_moments(model, tokenizer, total_samples, max_tokens,
                     batch_samples=None, actual_tokens_hint=None, seed=None):
    """Computes sample variance and excess kurtosis."""
    world_size = ddp_world_size()
    num_samples = (total_samples + world_size - 1) // world_size
    n = num_samples * world_size
    magnet = sample_magnets(model, tokenizer, num_samples, max_tokens, batch_samples=batch_samples, seed=seed)
    normalized_magnet = magnet / math.sqrt(actual_tokens_hint or max_tokens)
    total_normalized_magnet = torch.sum(normalized_magnet)
    if world_size > 1:
        dist.all_reduce(total_normalized_magnet, op=dist.ReduceOp.SUM)
    mean_normalized_magnet = total_normalized_magnet / n
    normalized_magnet_var = torch.sum((normalized_magnet - mean_normalized_magnet) ** 2)
    normalized_magnet_fourth = torch.sum((normalized_magnet - mean_normalized_magnet) ** 4)
    if world_size > 1:
        dist.all_reduce(normalized_magnet_var, op=dist.ReduceOp.SUM)
        dist.all_reduce(normalized_magnet_fourth, op=dist.ReduceOp.SUM)
    biased_var = normalized_magnet_var / n
    biased_fourth = normalized_magnet_fourth / n
    unbiased_var = normalized_magnet_var / (n - 1)
    sample_kurtosis = biased_fourth / biased_var ** 2 - 3
    fisher_kurtosis = (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * sample_kurtosis + 6)
    return unbiased_var.item(), fisher_kurtosis.item()
