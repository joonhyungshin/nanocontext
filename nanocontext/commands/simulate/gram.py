from collections import deque

import click
import numpy as np

from nanocontext.commands.common import echo
from nanocontext.tree import IsingBroadcastChannel, PerfectTreeConfig, BroadcastForest
from nanocontext.utils import RNGManager, d_order


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
    perplexity = compute_perplexity(forest, channel, context_len)
    echo(f"Perplexity: {np.mean(perplexity).item()}")


class BPNode:
    def __init__(self, height, rho, value=None):
        self.height = height
        self.rho = rho
        self.children = deque()
        self.parent = None
        self.value = value
        self.total_belief = 0
        self.incoming_msg = 0
        self.outgoing_msg = 0

    @property
    def num_children(self):
        return len(self.children)

    @property
    def is_leaf(self):
        return self.num_children == 0

    @property
    def is_root(self):
        return self.parent is None

    @property
    def first_child(self):
        return self.children[0]

    @property
    def last_child(self):
        return self.children[-1]

    @property
    def prob(self):
        return (self.total_belief + 1) / 2

    def add_child(self, value=None):
        child = BPNode(self.height - 1, self.rho, value=value)
        child.parent = self
        self.children.append(child)
        return child

    def pop_child(self):
        child = self.children.popleft()
        child.parent = None
        return child

    def create_parent(self, value=None):
        self.parent = BPNode(self.height + 1, self.rho, value=value)
        self.parent.children.append(self)
        return self.parent

    def update_belief(self):
        if self.value is not None:
            self.total_belief = self.value
            self.outgoing_msg = np.atanh(self.rho * self.value)
            for child in self.children:
                child.incoming_msg = self.outgoing_msg
        else:
            msg = 0
            msg += self.incoming_msg
            for child in self.children:
                msg += child.outgoing_msg
            self.total_belief = np.tanh(msg)
            self.outgoing_msg = np.atanh(self.rho * np.tanh(msg - self.incoming_msg))
            for child in self.children:
                child.incoming_msg = np.atanh(self.rho * np.tanh(msg - child.outgoing_msg))


def compute_perplexity(forest: BroadcastForest, channel: IsingBroadcastChannel, context_len):
    d, height = forest.d, forest.height
    rho = channel.rho
    h_profile = [0] * height + [1]
    h_inexact = height
    perplexity = np.zeros(forest.num_trees)
    node = BPNode(1, rho)
    num_tokens = d ** (height - 1) * (d + 1)
    for token_idx in range(1, num_tokens):
        if token_idx > context_len:
            popped_idx = token_idx - context_len - 1
            if popped_idx % (d + 1) == 0:
                h = d_order(popped_idx // (d + 1), d) + 1 if popped_idx else height
                if h >= h_inexact:
                    assert h_profile[h] > 0
                    h_profile[h] -= 1
                    while h > 0 and h_profile[h] == 0:
                        h_inexact = min(h_inexact, h - 1)
                        h -= 1
            else:
                while not node.is_root:
                    node = node.parent
                while node.height > 1:
                    node = node.first_child
                while not node.is_root:
                    if node.first_child.is_leaf:
                        node.pop_child()
                    node.update_belief()
                    node = node.parent
                if node.first_child.is_leaf:
                    node.pop_child()
                node.update_belief()
                while node.height > 1:
                    node = node.last_child
                    node.update_belief()
                if h_inexact == 0:
                    assert h_profile[0] > 0
                    h_profile[0] -= 1

        # Get next token
        if token_idx % (d + 1) == 0:
            punctuation = d_order(token_idx // (d + 1), d) + 1
            spins = None
        else:
            punctuation = 0
            leaf_idx = token_idx // (d + 1) * d + (token_idx % (d + 1) - 1)
            spins = forest.get_values(height, leaf_idx)

        # Compute perplexity
        if spins is not None:
            prob = (node.total_belief * spins * rho + 1) / 2
            if h_inexact == 0:
                prob *= 1 - 1 / (d - h_profile[0] + 1)
        else:
            prob = 1
            for i in range(h_inexact, punctuation):
                max_level = d if i > 0 else d + 1
                prob *= 1 / (max_level - h_profile[i])
            if h_inexact <= punctuation:
                prob *= 1 - 1 / (d - h_profile[punctuation])
        perplexity += np.log(1 / prob)

        # Add node
        if spins is None:
            node.update_belief()
            for _ in range(punctuation):
                if node.is_root:
                    node.create_parent()
                node = node.parent
                node.update_belief()
            for _ in range(punctuation):
                child = node.add_child()
                node.update_belief()
                node = child
            node.update_belief()
            h_inexact = max(h_inexact, punctuation)
            h_profile[punctuation] += 1
            for i in range(punctuation):
                h_profile[i] = 0
        else:
            child = node.add_child(spins)
            child.update_belief()
            node.update_belief()
            h_profile[0] += 1
    return perplexity
