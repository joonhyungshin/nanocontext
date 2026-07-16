from collections import deque, namedtuple
from enum import Enum

import numpy as np

from nanocontext.tree import IsingBroadcastChannel, BroadcastForest, PerfectTreeConfig
from nanocontext.utils import d_order


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


class TokenType(Enum):
    SPIN = 0
    PUNCTUATION = 1

Token = namedtuple("Token", ["type", "value"])


class BPSimulator:
    def __init__(self, d, rho, num_samples=1, seed=None):
        self.d = d
        self.rho = rho
        self.num_samples = num_samples
        self.rng = np.random.default_rng(seed)
        self.h_exact = 0
        self.h_profile = {}
        self._node = None

    @property
    def node(self):
        if self._node is None:
            raise ValueError("Simulator has not yet been initialized")
        return self._node

    def initialize(self):
        self._node = BPNode(1, rho=self.rho)
        self.h_exact = -1
        self.h_profile = {}

    def cross_entropy(self, token: Token):
        if token.type == TokenType.SPIN:
            spins = token.value
            prob = (self.node.total_belief * spins * self.rho + 1) / 2
            if self.h_exact == 0:
                prob *= 1 - 1 / (self.d - self.h_profile.get(0, 0) + 1)
        elif token.type == TokenType.PUNCTUATION:
            prob = 1
            punctuation = token.value
            if self.h_exact >= 0:
                for i in range(self.h_exact, punctuation):
                    max_level = self.d if i > 0 else self.d + 1
                    prob *= 1 / (max_level - self.h_profile.get(i, 0))
                if self.h_exact <= punctuation:
                    prob *= 1 - 1 / (self.d - self.h_profile.get(punctuation, 0))
        else:
            return np.inf
        return np.log(1 / prob)

    def observe(self, token: Token):
        if token.type == TokenType.SPIN:
            spins = token.value
            child = self.node.add_child(spins)
            child.update_belief()
            self.node.update_belief()
            self.h_profile[0] = self.h_profile.get(0, 0) + 1
        elif token.type == TokenType.PUNCTUATION:
            punctuation = token.value
            self.node.update_belief()
            for _ in range(punctuation):
                if self.node.is_root:
                    self.node.create_parent()
                self._node = self.node.parent
                self.node.update_belief()
            for _ in range(punctuation):
                child = self.node.add_child()
                self.node.update_belief()
                self._node = child
            self.node.update_belief()
            if 0 <= self.h_exact < punctuation:
                self.h_exact = punctuation
            self.h_profile[punctuation] = self.h_profile.get(punctuation, 0) + 1
            for i in range(punctuation):
                self.h_profile.pop(i, None)

def compute_perplexity(leaves: np.ndarray, config: PerfectTreeConfig,
                       channel: IsingBroadcastChannel, context_len):
    d, height = config.d, config.height
    num_trees, num_leaves = leaves.shape
    rho = channel.rho
    h_profile = [0] * height + [1]
    h_inexact = height
    perplexity = np.zeros(num_trees)
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
            spins = leaves[:, leaf_idx]

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
