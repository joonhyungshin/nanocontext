from collections import deque

import numpy as np
import torch

from nanocontext.utils import d_order


class BroadcastTree:
    class Node:
        def __init__(self, value=None):
            self.parent = None
            self.children = []
            self.value = value

        def add_child(self, child):
            self.children.append(child)
            child.parent = self

        def create_child(self, value=None):
            child = BroadcastTree.Node(value)
            self.add_child(child)
            return child

        def get_parent_or_create(self):
            created = False
            if self.parent is None:
                self.parent = BroadcastTree.Node()
                self.parent.add_child(self)
                created = True
            return self.parent, created

    def __init__(self, rho, seed=None):
        self.rho = rho
        self.rng = np.random.default_rng(seed)
        self.root = self.Node()

    def _draw_node(self, node, canvas, canvas_idx, depth):
        canvas[depth] += " " * (canvas_idx - len(canvas[depth]))
        if node.value is None:
            canvas[depth] += "#"
        elif node.value < 0:
            canvas[depth] += "-"
        else:
            canvas[depth] += "+"
        if node.children:
            if len(canvas) == depth + 1:
                canvas.append("")
            for child in node.children:
                canvas_idx = self._draw_node(child, canvas, canvas_idx, depth + 1)
        else:
            canvas_idx += 1
        return canvas_idx

    def print_tree(self):
        canvas = [""]
        self._draw_node(self.root, canvas, 0, 0)
        for line in canvas:
            print(line)


class RegularBroadcastTree:
    def __init__(self, d, rho, height, root_prob=None, seed=None):
        self.d = d
        self.rho = rho
        self.height = height
        self.root_prob = root_prob if root_prob is not None else [0.5, 0.5]
        self.values = []
        self.rng = np.random.default_rng(seed)

    def sample(self):
        self.values = [self.rng.choice([-1, 1], size=(1,), p=self.root_prob)]
        cur_size = 1
        flip_prob = [(1 - self.rho) / 2, (1 + self.rho) / 2]
        for i in range(self.height):
            flip = self.rng.choice([-1, 1], size=(self.d, cur_size), p=flip_prob)
            self.values.append((flip * self.values[-1]).ravel(order='F'))
            cur_size *= self.d

    @property
    def sampled(self):
        return len(self.values) == self.height + 1

    @property
    def leaves(self):
        return self.values[-1] if self.sampled else None

    @property
    def root(self):
        return self.values[0][0] if self.sampled else None

    def ancestors(self, leaf_idx):
        if not self.sampled:
            return None
        seq = []
        for i in range(self.height, -1, -1):
            seq.append(self.values[leaf_idx])
            leaf_idx //= self.d
        seq.reverse()
        return seq

    def print_tree(self):
        if not self.sampled:
            return
        msg = ""
        for i, layer in enumerate(self.values):
            msg += (" " * (self.d ** (self.height - i) - 1)).join([("+" if node > 0 else "-")
                                                                   for node in layer])
            msg += "\n"
        print(msg)


class BroadcastTreeTokenizer:
    def __init__(self, max_vocab_size):
        self.max_vocab_size = max_vocab_size

    def tokenize(self, tree):
        tokens = []
        for idx, spin in enumerate(tree.leaves):
            if idx > 0:
                zero_cnt = d_order(idx, tree.d)
                if zero_cnt > 0:
                    tokens.append(min(zero_cnt + 2, self.max_vocab_size))
            tokens.append(1 if spin < 0 else 2)
        return tokens


def dynamic_broadcast_tree(d, rho, height, batch_height, seed=None):
    ancestors = []  # INORDER traversal of tree
    leaf_idx = 0
    batch_height = min(height, batch_height)
    batch_len = d**batch_height
    sibling_indices = []
    rng = np.random.default_rng(seed)
    flip_prob = [(1 - rho) / 2, (1 + rho) / 2]
    while len(ancestors) != height - batch_height + 1:
        rho_flip = ancestors[-1] * rho if ancestors else 0
        root_prob = [(1 - rho_flip) / 2, (1 + rho_flip) / 2]
        tree = RegularBroadcastTree(d, rho, batch_height, root_prob=root_prob, seed=rng)
        tree.sample()
        yield leaf_idx, tree, ancestors.copy()
        target_idx = len(ancestors) - 1
        while target_idx >= 0 and sibling_indices[target_idx] == d - 1:
            sibling_indices[target_idx] = 0
            target_idx -= 1
        if target_idx == -1:
            new_root = (ancestors[0] if ancestors else tree.root) * rng.choice([-1, 1], p=flip_prob)
            ancestors = [new_root] + ancestors
            sibling_indices = [0] + sibling_indices
            target_idx = 0
        sibling_indices[target_idx] += 1
        target_idx += 1
        while target_idx < len(ancestors):
            ancestors[target_idx] = ancestors[target_idx - 1] * rng.choice([-1, 1], p=flip_prob)
            target_idx += 1
        leaf_idx += batch_len


def tokenized_broadcast_trees(d, rho, height, batch_height, max_vocab_size=32, seed=None):
    rng = np.random.default_rng(seed)
    tokenizer = BroadcastTreeTokenizer(max_vocab_size - 1)
    while True:
        tree = dynamic_broadcast_tree(d, rho, height, batch_height, seed=rng)
        for leaf_idx, subtree, ancestors in tree:
            tokens = []
            if leaf_idx == 0:
                tokens.append(0)
            else:
                zero_cnt = d_order(leaf_idx, d)
                tokens.append(zero_cnt + 2)
            tokens.extend(tokenizer.tokenize(subtree))
            yield tokens


def broadcast_tree_data_loader(d, rho, height, batch_size, seq_len, batch_height, max_vocab_size=32,
                               bos_token=None, device="cpu", seed=None):
    token_buffer = deque()
    needed_tokens = batch_size * seq_len + 1
    rng = np.random.default_rng(seed)
    trees = tokenized_broadcast_trees(d, rho, height, batch_height, max_vocab_size, rng)
    while True:
        while len(token_buffer) < needed_tokens:
            if bos_token is not None:
                token_buffer.append(bos_token)
                token_buffer.extend(next(trees))
        tokens = [token_buffer.popleft() for _ in range(needed_tokens)]
        use_cuda_opt = device == "cuda"
        scratch = torch.tensor(tokens, dtype=torch.long, pin_memory=use_cuda_opt)
        inputs_cpu = scratch[:-1]
        targets_cpu = scratch[1:]
        inputs = inputs_cpu.view(batch_size, seq_len).to(device=device, non_blocking=use_cuda_opt)
        targets = targets_cpu.view(batch_size, seq_len).to(device=device, non_blocking=use_cuda_opt)
        yield inputs, targets


def decode_trees(tokens, rho=None, seed=None):
    trees = []
    current_tree = None
    current_node = None
    rng = np.random.default_rng(seed)
    for token in tokens:
        if token == 0:
            if current_tree is not None:
                trees.append(current_tree)
            current_tree = BroadcastTree(rho, seed=rng)
            current_node = current_tree.root
        elif token in [1, 2]:
            spin = -1 if token == 1 else 1
            if current_tree is None:
                raise ValueError("invalid tokens: expected 0 in the beginning")
            current_node.create_child(spin)
        elif token >= 3:
            jump_height = token - 2
            for i in range(jump_height):
                current_node, created = current_node.get_parent_or_create()
                if created:
                    current_tree.root = current_node
            for i in range(jump_height):
                current_node = current_node.create_child()
    trees.append(current_tree)
    return trees
