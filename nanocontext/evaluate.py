import torch
import torch.distributed as dist

from .data.broadcast_tree import decode_trees
from .sample import NanochatSampler
from .utils import ddp_world_size


@torch.inference_mode()
def evaluate_var_sum(model, num_samples, max_tokens, seed=None):
    world_size = ddp_world_size()
    total_samples = num_samples * world_size
    sampler = NanochatSampler(model, seed=seed)
    tokens = sampler.generate_batch([0], num_samples=num_samples, max_tokens=max_tokens)
    spin_sum = torch.zeros(num_samples, dtype=torch.int64, device=model.device)
    mean_spin_sum = torch.tensor(0, dtype=torch.int64, device=model.device)
    for i, tree_tokens in enumerate(tokens):
        if len(tree_tokens) <= 1:
            continue
        for tree in decode_trees(tree_tokens):
            spin_sum[i] += sum(tree.get_leaves_values())
        mean_spin_sum += spin_sum[i]
    if world_size > 1:
        dist.all_reduce(mean_spin_sum, op=dist.ReduceOp.SUM)
    mean_spin_sum = mean_spin_sum / total_samples
    spin_var = torch.sum((spin_sum - mean_spin_sum) ** 2)
    if world_size > 1:
        dist.all_reduce(spin_var, op=dist.ReduceOp.SUM)
    spin_var = spin_var / (total_samples - 1)
    return spin_var.item()
