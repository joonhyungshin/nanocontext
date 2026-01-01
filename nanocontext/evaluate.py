import torch
import torch.distributed as dist

from .sample import NanochatSampler
from .utils import ddp_world_size


@torch.inference_mode()
def evaluate_var_sum(d, model, num_samples, seed=None):
    sampler = NanochatSampler(model, seed=seed)
    tokens = [0]
    for token in sampler.generate([0], num_samples=num_samples, max_tokens=16, end_token=0):
        tokens.append(token[0])
    if tokens[-1] == 0:
        tokens = tokens[:-1]
    if len(tokens) <= 1:
        echo("(empty)")
    else:
        echo(decode_trees(tokens)[0])
