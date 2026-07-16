import click
import numpy as np
import torch
import torch.distributed as dist

from nanocontext.data.broadcast_tree import load_engine
from nanocontext.evaluate.ising import evaluate_perplexity
from nanocontext.evaluate.bp import compute_perplexity
from nanocontext.tree import IsingBroadcastChannel, PerfectTreeConfig
from nanocontext.tree.coloring import ColoringSpace
from nanocontext.tree.ising import IsingSpace
from nanocontext.utils import ddp_context, device_to_use, RNGManager, ddp_world_size

from ..common import echo, make_prompt, get_max_tokens, display_recon_stat


@click.command()
@click.option("-d", help="number of children of a tree", type=int, required=True)
@click.option("--height", help="height of a tree", type=int, required=True)
@click.option("--temperature", help="sampling temperature", default=1.0, type=float)
@click.option("--top-k", help="top-k sampling", type=int)
@click.option("--rho", help="correlation for Ising experiment", type=float)
@click.option("--samples", help="number of samples to generate", default=1024, type=int)
@click.option("--sample-batch", help="batch size for sampling", type=int)
@click.option("--model-path", help="path to model", type=str, required=True)
@click.option("--seed", help="random seed", type=int)
def evaluate(d, height, rho, samples, sample_batch, model_path,
             temperature, top_k, seed):
    rng = RNGManager(seed=seed)
    echo(f"evaluating with seed: {rng.seed}")
    echo(f"using model: {model_path}")
    tree_conf = PerfectTreeConfig(d=d, height=height)
    sample_batch = sample_batch or samples

    with ddp_context():
        device = device_to_use()
        engine = load_engine(model_path, device, seed=rng.local_torch_rng(device))
        engine.model.eval()
        value_space = engine.tokenizer.value_space
        prompt = make_prompt(engine.tokenizer, tree_conf)
        max_tokens = get_max_tokens(d, height)

        if isinstance(value_space, IsingSpace):
            echo(f"generating {samples} samples...")
            if rho is None:
                raise ValueError("rho must be provided in Ising experiment to evaluate entropy.")
            channel = IsingBroadcastChannel(rho=rho, seed=rng.local_numpy_rng)
            world_size = ddp_world_size()
            num_samples = (samples + world_size - 1) // world_size
            samples = num_samples * world_size
            leaves = torch.randint(0, 2, (num_samples, d ** height),
                                   device=device, generator=rng.local_torch_rng(device)) * 2 - 1
            for i in range(0, num_samples, sample_batch):
                actual_batch = min(num_samples - i, sample_batch)
                gen_kwargs = dict(num_samples=actual_batch, temperature=temperature, top_k=top_k, max_context_tokens=64)
                tree_generator = engine.generate_patched_tree(prompt, max_tokens, tree_conf, allow_many=True, **gen_kwargs)
                for (tree_idx, tree) in enumerate(tree_generator):
                    for (leaf_idx, leaf) in enumerate(tree.leaves_values_stream()):
                        if leaf is not None:
                            leaves[tree_idx, leaf_idx] = leaf
            if world_size > 1:
                all_leaves = torch.empty((samples, d ** height), device=device)
                dist.all_gather_into_tensor(all_leaves, leaves)
            else:
                all_leaves = leaves

            perplexity = compute_perplexity(all_leaves.detach().cpu().numpy(), tree_conf, channel, max_tokens * 2)
            echo(f"Cross entropy: {np.mean(perplexity).item()}")
        elif isinstance(value_space, ColoringSpace):
            raise click.ClickException("Reverse KL evaluation only supports Ising model at this time")


def gather_forest(engine, total_samples, batch_samples):
    world_size = ddp_world_size()
    num_samples = (total_samples + world_size - 1) // world_size
    total_samples = num_samples * world_size
    for i in range(0, num_samples, batch_samples):
        actual_batch_samples = min(num_samples - i, batch_samples)
