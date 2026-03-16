import torch
import torch.distributed as dist

from nanocontext.tree import LinkedOrderedTree, ValueDomain, PerfectTreeConfig
from nanocontext.data.broadcast_tree import Engine
from nanocontext.utils import ddp_world_size


class UnsatisfiedException(Exception):
    def __init__(self, depth):
        super().__init__()
        self.depth = depth


class InvalidStructureException(Exception):
    pass


def get_color_constraint(node: LinkedOrderedTree.Node, domain: ValueDomain, config: PerfectTreeConfig, depth):
    if node.value is not None:
        return node.value
    colors = set(range(domain.get_size()))
    if depth == config.height and len(node.children) != 0:
        raise InvalidStructureException()
    if depth < config.height and len(node.children) != config.d:
        raise InvalidStructureException()
    for child in node.children:
        child_constraint = get_color_constraint(child, domain, config, depth + 1)
        if child_constraint is not None:
            colors.discard(child_constraint)
    if len(colors) == 0:
        raise UnsatisfiedException(depth)
    elif len(colors) == 1:
        return colors.pop()
    return None


def get_root_constraint(tree: LinkedOrderedTree, domain: ValueDomain, config: PerfectTreeConfig):
    if tree.root is None:
        raise InvalidStructureException()
    return get_color_constraint(tree.root, domain, config, 0)


def check_validity(engine: Engine, prompt, total_samples, max_tokens, config: PerfectTreeConfig,
                   batch_samples=None):
    world_size = ddp_world_size()
    num_samples = (total_samples + world_size - 1) // world_size
    batch_samples = batch_samples or num_samples
    tokenizer = engine.tokenizer
    stat_tensor = torch.zeros(4, device=engine.device, dtype=torch.int64)
    unsat_tensor = torch.zeros(config.height, device=engine.device, dtype=torch.int64)
    for i in range(0, num_samples, batch_samples):
        actual_batch_samples = min(num_samples - i, batch_samples)
        trees = engine.generate_tree(prompt, num_samples=actual_batch_samples, max_tokens=max_tokens)
        for tree in trees:
            try:
                constraint = get_root_constraint(tree, tokenizer.domain, config)
                if constraint is None:
                    stat_tensor[3] += 1
                else:
                    stat_tensor[2] += 1
            except UnsatisfiedException as e:
                stat_tensor[0] += 1
                unsat_tensor[e.depth] += 1
            except InvalidStructureException:
                stat_tensor[1] += 1
    if world_size > 1:
        dist.all_reduce(stat_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(unsat_tensor, op=dist.ReduceOp.SUM)
    stat = {
        "unsatisfied": {
            "total": stat_tensor[0],
            "details": {depth: unsat_tensor[depth] for depth in range(config.height)},
        },
        "invalid": stat_tensor[1],
        "constrained": stat_tensor[2],
        "free": stat_tensor[3],
    }
    return stat
