import click
import numpy as np

from nanocontext.commands.common import echo
from nanocontext.tree import IsingBroadcastChannel, PerfectTreeConfig, BroadcastForest
from nanocontext.utils import RNGManager

from nanocontext.evaluate.bp import compute_perplexity


@click.command()
@click.option("-d", help="number of children of a tree", default=3, type=int)
@click.option("--rho", help="correlation for Ising experiment", type=float)
@click.option("--height", help="height of a tree", type=int, required=True)
@click.option("--context-size", "context_len", help="context length", default=2048, type=int)
@click.option("--samples", help="number of samples to generate", default=1024, type=int)
@click.option("--seed", help="random seed", type=int)
def gram(d, height, rho, context_len, samples, seed):
    rng = RNGManager(seed=seed)
    tree_conf = PerfectTreeConfig(d=d, height=height)
    echo(f"Simulating n-gram model with seed: {rng.seed}")
    echo(f"{d}-ary tree with height {height}")
    echo(f"Ising experiment with rho: {rho}")
    echo(f"Context length: {context_len}")

    channel = IsingBroadcastChannel(rho, seed=rng.global_numpy_rng)
    forest = BroadcastForest(tree_conf, channel, num_trees=samples)
    forest.sample()
    perplexity = compute_perplexity(forest.get_leaves(), tree_conf, channel, context_len)
    echo(f"Perplexity: {np.mean(perplexity).item()}")
