import torch
import torch.distributed as dist
from torch.distributions import Categorical

from nanocontext.evaluate import infer_summary_every
from nanocontext.tree import PerfectTreeConfig, AbstractOrderedTree
from nanocontext.data.broadcast_tree import Engine, StatefulEngine
from nanocontext.utils import ddp_world_size


class UnsatisfiedException(Exception):
    def __init__(self, depth):
        super().__init__()
        self.depth = depth


class InvalidStructureException(Exception):
    pass


def get_color_constraint(tree: AbstractOrderedTree, node, num_colors, depth, only_leaves=False):
    colors = set(range(num_colors))
    value = tree.get_value(node)
    if value is not None and (not only_leaves or tree.num_children(node) == 0):
        return value
    for child in tree.children_stream(node):
        child_constraint = get_color_constraint(tree, child, num_colors, depth + 1, only_leaves=only_leaves)
        if child_constraint is not None:
            colors.discard(child_constraint)
    if len(colors) == 0:
        raise UnsatisfiedException(depth)
    elif len(colors) == 1:
        return colors.pop()
    return None


def check_subtree_structure(tree: AbstractOrderedTree, node, config: PerfectTreeConfig, depth):
    num_children = tree.num_children(node)
    if depth == config.height and num_children != 0:
        raise InvalidStructureException()
    if depth < config.height and num_children != config.d:
        raise InvalidStructureException()
    for child in tree.children_stream(node):
        check_subtree_structure(tree, child, config, depth + 1)


def get_root_constraint(tree: AbstractOrderedTree, num_colors, only_leaves=False):
    root = tree.get_root()
    if root is None:
        raise InvalidStructureException()
    return get_color_constraint(tree, root, num_colors, 0, only_leaves=only_leaves)


def check_structure(tree: AbstractOrderedTree, config: PerfectTreeConfig):
    root = tree.get_root()
    if root is None:
        raise InvalidStructureException()
    check_subtree_structure(tree, root, config, 0)


def check_validity(engine: Engine, prompt, total_samples, max_tokens, config: PerfectTreeConfig,
                   batch_samples=None, patch=False, max_summary_tokens=64):
    world_size = ddp_world_size()
    num_samples = (total_samples + world_size - 1) // world_size
    batch_samples = batch_samples or num_samples
    tokenizer = engine.tokenizer
    stat_tensor = torch.zeros(4, device=engine.device, dtype=torch.int64)
    unsat_tensor = torch.zeros(config.height, device=engine.device, dtype=torch.int64)
    for i in range(0, num_samples, batch_samples):
        actual_batch_samples = min(num_samples - i, batch_samples)
        if patch:
            trees = engine.generate_patched_tree(prompt, max_tokens, config,
                                                 num_samples=actual_batch_samples,
                                                 max_context_tokens=max_summary_tokens,
                                                 allow_many=True)
        else:
            trees = engine.generate_tree(prompt, max_tokens, num_samples=actual_batch_samples,
                                         max_context_tokens=max_summary_tokens, allow_many=False)
        for tree in trees:
            try:
                check_structure(tree, config)
                constraint = get_root_constraint(tree, tokenizer.value_space.get_size())
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


@torch.inference_mode()
def evaluate_entropy(engine: Engine, prompt, total_samples, max_tokens, tree_config: PerfectTreeConfig,
                     batch_samples=None):
    sampler = engine.sampler
    world_size = ddp_world_size()
    num_samples = (total_samples + world_size - 1) // world_size
    batch_samples = batch_samples or num_samples
    total_samples = num_samples * world_size
    entropy = torch.tensor([0], device=engine.device, dtype=torch.float)

    def logits_stream(n):
        if isinstance(engine, StatefulEngine):
            summary_len = engine.summary_len or len(prompt)
            content_len = engine.content_len or infer_summary_every(engine, prompt, tree_config)
            current_prompt = prompt
            while True:
                current_tokens = 0
                new_prompt = torch.empty((n, summary_len), device=sampler.device, dtype=torch.long)
                for t, l in sampler.stream(current_prompt, num_samples=n):
                    yield l
                    if current_tokens >= content_len:
                        new_prompt[:, current_tokens - content_len] = t
                    current_tokens += 1
                    if current_tokens >= content_len + summary_len:
                        break
                current_prompt = new_prompt
        else:
            for _, l in sampler.stream(prompt, num_samples=n):
                yield l

    for i in range(0, num_samples, batch_samples):
        actual_batch_samples = min(num_samples - i, batch_samples)
        num_tokens = 0
        for logits in logits_stream(actual_batch_samples):
            law = Categorical(logits=logits)
            entropy += law.entropy().sum()
            num_tokens += 1
            if num_tokens >= max_tokens:
                break
    if world_size > 1:
        dist.all_reduce(entropy, op=dist.ReduceOp.SUM)
    entropy = entropy.item() / total_samples
    return entropy
