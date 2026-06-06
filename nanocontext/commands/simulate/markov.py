import math

import click
import numpy as np

from nanocontext.commands.common import echo
from nanocontext.evaluate.coloring import get_root_constraint, UnsatisfiedException
from nanocontext.evaluate.ising import compute_moments
from nanocontext.tree import IsingBroadcastChannel, ColoringBroadcastChannel, PerfectTreeConfig, InferenceTree
from nanocontext.tree.broadcast import markov_forest
from nanocontext.utils import compute_moments, RNGManager



@click.command()
@click.option("-d", help="number of children of a tree", default=3, type=int)
@click.option("--rho", help="correlation for Ising experiment", type=float)
@click.option("-k", "--colors", "k", help="number of colors for coloring experiment", type=int)
@click.option("--height", help="height of a tree", type=int, required=True)
@click.option("--samples", help="number of samples to generate", default=1024, type=int)
@click.option("--markov-height", help="height when sampled using the Markov process", type=int)
@click.option("--seed", help="random seed", type=int)
def markov(d, height, rho, k, samples, markov_height, seed):
    if (rho is None and k is None) or not (rho is None or k is None):
        raise ValueError("exactly one of k (coloring) or rho (Ising) must be given")
    rng = RNGManager(seed=seed)
    tree_conf = PerfectTreeConfig(d=d, height=height)
    markov_height = min(markov_height, height) if markov_height is not None else height
    markov_kwargs = dict(batch_height=markov_height, num_trees=samples, seed=rng.global_numpy_rng)
    echo(f"Simulating Markov model with seed: {rng.seed}")
    echo(f"{d}-ary tree with height {height} and batch height {markov_height}")

    if rho is not None:
        echo(f"Ising experiment with rho: {rho}")
        channel = IsingBroadcastChannel(rho, seed=rng.global_numpy_rng)
        magnet = np.zeros(samples)
        for forest in markov_forest(tree_conf, channel, **markov_kwargs):
            magnet += np.sum(forest.values[-1], axis=1) / math.sqrt(d ** height)
        var, kurtosis = compute_moments(magnet)
        echo(f"Variance: {var}")
        echo(f"Kurtosis: {kurtosis}")
    else:
        echo(f"Coloring experiment with k: {k}")
        channel = ColoringBroadcastChannel(k, seed=rng.global_numpy_rng)
        free_count = 0
        constrained_count = 0
        unsat_total_count = 0
        unsat_count = {i: 0 for i in range(height - markov_height)}
        leaves = np.empty((samples, d ** height))
        batch_leaves = d ** markov_height
        for j, forest in enumerate(markov_forest(tree_conf, channel, **markov_kwargs)):
            leaves[:, j * batch_leaves:(j + 1) * batch_leaves] = forest.values[-1]
        for i in range(samples):
            tree = InferenceTree(tree_conf, leaves[i])
            try:
                constraint = get_root_constraint(tree, k)
                if constraint is None:
                    free_count += 1
                else:
                    constrained_count += 1
            except UnsatisfiedException as e:
                unsat_total_count += 1
                unsat_count[e.depth] += 1
        echo(f"Free trees: {free_count}")
        echo(f"Constrained trees: {constrained_count}")
        echo(f"Unsatisfied trees: {unsat_total_count}")
        if unsat_total_count > 0:
            for i in range(height - markov_height):
                echo(f"  at depth {i}: {unsat_count[i]}")
        echo(f"Valid rate: {(free_count + constrained_count) / samples}")
