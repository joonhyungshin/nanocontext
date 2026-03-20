import math

import torch
import torch.distributed as dist

from nanocontext.data.broadcast_tree import Engine
from nanocontext.utils import ddp_world_size


@torch.inference_mode()
def sample_magnets(engine: Engine, prompt, num_samples, max_tokens,
                   batch_samples=None):
    batch_samples = batch_samples or num_samples
    magnet = torch.empty(num_samples, device=engine.device)
    tokenizer = engine.tokenizer
    pos_token = tokenizer.tokenize_value(1)
    neg_token = tokenizer.tokenize_value(-1)
    for i in range(0, num_samples, batch_samples):
        actual_batch_samples = min(num_samples - i, batch_samples)
        for token_tensor in engine.generate_tree_tokens_tensor_stream(prompt,
                                                                      max_tokens=max_tokens,
                                                                      allow_many=True,
                                                                      num_samples=actual_batch_samples):
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


def evaluate_moments(engine: Engine, prompt, total_samples, max_tokens,
                     batch_samples=None, actual_tokens_hint=None):
    """Computes sample variance and excess kurtosis."""
    world_size = ddp_world_size()
    num_samples = (total_samples + world_size - 1) // world_size
    n = num_samples * world_size
    magnet = sample_magnets(engine, prompt, num_samples, max_tokens, batch_samples=batch_samples)
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
