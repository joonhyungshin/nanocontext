import math

import torch
import torch.distributed as dist

from .sample import NanochatSampler
from .utils import ddp_world_size


@torch.inference_mode()
def evaluate_moments(model, tokenizer, num_samples, max_tokens, actual_tokens_hint=None, seed=None):
    """Computes sample variance and excess kurtosis."""
    world_size = ddp_world_size()
    n = num_samples * world_size
    sampler = NanochatSampler(model, seed=seed)
    tokens = sampler.generate_batch_tensor([0], num_samples=num_samples, max_tokens=max_tokens)
    spin_sum = torch.sum(tokens == tokenizer.pos_token, dim=1) - torch.sum(tokens == tokenizer.neg_token, dim=1)
    normalized_spin_sum = spin_sum / math.sqrt(actual_tokens_hint or max_tokens)
    total_normalized_spin_sum = torch.sum(normalized_spin_sum)
    if world_size > 1:
        dist.all_reduce(total_normalized_spin_sum, op=dist.ReduceOp.SUM)
    mean_normalized_spin_sum = total_normalized_spin_sum / n
    normalized_spin_var = torch.sum((normalized_spin_sum - mean_normalized_spin_sum) ** 2)
    normalized_spin_fourth = torch.sum((normalized_spin_sum - mean_normalized_spin_sum) ** 4)
    if world_size > 1:
        dist.all_reduce(normalized_spin_var, op=dist.ReduceOp.SUM)
        dist.all_reduce(normalized_spin_fourth, op=dist.ReduceOp.SUM)
    biased_var = normalized_spin_var / n
    biased_fourth = normalized_spin_fourth / n
    unbiased_var = normalized_spin_var / (n - 1)
    sample_kurtosis = biased_fourth / biased_var ** 2 - 3
    fisher_kurtosis = (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * sample_kurtosis + 6)
    return unbiased_var.item(), fisher_kurtosis.item()
