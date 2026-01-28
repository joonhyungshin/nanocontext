from dataclasses import dataclass

import numpy as np

from nanocontext.utils import d_order


class OrderedTree:
    class Node:
        def __init__(self, value=None):
            self.parent = None
            self.children = []
            self.value = value

        def add_child(self, child):
            self.children.append(child)
            child.parent = self

        def create_child(self, value=None):
            child = OrderedTree.Node(value)
            self.add_child(child)
            return child

        def get_parent_or_create(self):
            created = False
            if self.parent is None:
                self.parent = OrderedTree.Node()
                self.parent.add_child(self)
                created = True
            return self.parent, created

        def traverse(self):
            yield self
            for child in self.children:
                yield from child.traverse()

    def is_singleton(self):
        return len(self.root.children) == 0

    def __init__(self):
        self.root = self.Node()

    def get_leaves(self):
        for node in self.root.traverse():
            if not node.children and node.value is not None:
                yield node.value

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

    def __str__(self):
        canvas = [""]
        self._draw_node(self.root, canvas, 0, 0)
        return "\n".join(canvas)


@dataclass
class BroadcastConfig:
    d: int
    rho: float
    height: int


class AbstractBroadcastForest:
    def __init__(self, config: BroadcastConfig, seed=None):
        self.config = config
        self.rng = np.random.default_rng(seed)

    @property
    def d(self):
        return self.config.d

    @property
    def rho(self):
        return self.config.rho

    @property
    def height(self):
        return self.config.height


class BroadcastForest(AbstractBroadcastForest):
    def __init__(self, config: BroadcastConfig, num_trees=1, root_prob=None, seed=None):
        super().__init__(config, seed=seed)
        self.root_prob = root_prob if root_prob is not None else [0.5, 0.5]
        self.num_trees = num_trees
        self.values = []

    def _sample_roots(self):
        root_prob_arr = np.array(self.root_prob)
        if root_prob_arr.ndim == 1 and root_prob_arr.shape[0] == 2:
            self.values = [self.rng.choice([-1, 1], size=(self.num_trees, 1), p=self.root_prob)]
        else:
            assert root_prob_arr.ndim == 2 and root_prob_arr.shape[0] == 2
            p = root_prob_arr[1, :, np.newaxis]
            mask = self.rng.random((self.num_trees, 1)) < p
            self.values = [2 * mask.astype(int) - 1]

    def sample(self):
        self._sample_roots()
        cur_size = 1
        flip_prob = [(1 - self.rho) / 2, (1 + self.rho) / 2]
        for i in range(self.height):
            flip = self.rng.choice([-1, 1], size=(self.num_trees, cur_size, self.d), p=flip_prob)
            self.values.append((flip * self.values[-1][:, :, np.newaxis]).reshape((self.num_trees, -1)))
            cur_size *= self.d

    @property
    def sampled(self):
        return len(self.values) == self.height + 1

    @property
    def roots(self):
        return self.values[0][:, 0] if self.sampled else None

    @property
    def all_leaves(self):
        return self.values[-1] if self.sampled else None

    def ancestors(self, tree_idx, leaf_idx):
        if not self.sampled:
            return None
        seq = []
        for i in range(self.height, -1, -1):
            seq.append(self.values[i][tree_idx, leaf_idx])
            leaf_idx //= self.d
        seq.reverse()
        return seq

    def __str__(self):
        if not self.sampled:
            return "(not sampled)"
        msg = ""
        for j in range(self.num_trees):
            for i, layer in enumerate(self.values):
                msg += (" " * (self.d ** (self.height - i) - 1)).join([("+" if node > 0 else "-")
                                                                       for node in layer[j]])
                msg += "\n"
        return msg


class BroadcastTree(BroadcastForest):
    def __init__(self, config: BroadcastConfig, root_prob=None, seed=None):
        super().__init__(config, num_trees=1, root_prob=root_prob, seed=seed)

    def values_at(self, depth: int):
        return self.values[depth][0, :] if self.sampled else None

    @property
    def root(self):
        return self.values_at(0)[0]

    @property
    def num_leaves(self):
        return self.d ** self.height

    def get_leaves(self):
        return self.values_at(-1)

    def get_ancestors(self, leaf_idx):
        ancestors = []
        leaf_idx //= self.d
        for i in range(self.height):
            depth = self.height - i - 1
            ancestors.append(self.values_at(depth)[leaf_idx])
            leaf_idx //= self.d
        ancestors.reverse()
        return ancestors

    def summarize(self, start, end):
        idx = start
        while idx < end:
            segment_depth = self.height
            segment_len = 1
            segment_idx = idx
            while segment_idx % self.d == 0 and idx + segment_len * self.d <= end:
                segment_depth -= 1
                segment_len *= self.d
                segment_idx //= self.d
            segment_value = self.values_at(segment_depth)[segment_idx]
            yield segment_depth, segment_value
            idx += segment_len


class LazyBroadcastTree(AbstractBroadcastForest):
    def __init__(self, config: BroadcastConfig, root_prob=None, seed=None):
        super().__init__(config, seed=seed)
        self.root_prob = root_prob if root_prob is not None else [0.5, 0.5]
        self.rng = np.random.default_rng(seed)
        self.sampled_values = [{} for _ in range(self.height + 1)]
        self.sampled_subtrees = [{} for _ in range(self.height + 1)]

    @property
    def num_leaves(self):
        return self.d ** self.height

    def _sample(self, depth, idx):
        assert 0 <= idx < self.d ** depth
        assert depth == 0 or idx // self.d in self.sampled_values[depth - 1]
        assert idx not in self.sampled_values[depth]
        if depth == 0:
            self.sampled_values[depth][idx] = self.rng.choice([-1, 1], p=self.root_prob)
        else:
            flip_prob = [(1 - self.rho) / 2, (1 + self.rho) / 2]
            flip = self.rng.choice([-1, 1], p=flip_prob)
            self.sampled_values[depth][idx] = flip * self.get_value_or_sample(depth - 1, idx // self.d)
        return self.sampled_values[depth][idx]

    def is_sampled(self, depth, idx):
        assert 0 <= idx < self.d ** depth
        if idx in self.sampled_values[depth]:
            return True
        while depth >= 0:
            if idx in self.sampled_subtrees[depth]:
                return True
            depth -= 1
            idx //= self.d
        return False

    def get_value_or_sample(self, depth, idx):
        assert 0 <= idx < self.d ** depth
        if idx in self.sampled_values[depth]:
            return self.sampled_values[depth][idx]
        if self.is_sampled(depth, idx):
            raise ValueError("subtree already sampled in this position")
        ancestors = []
        ancestor_idx = idx // self.d
        for i in range(depth):
            ancestor_depth = depth - 1 - i
            if ancestor_idx in self.sampled_values[ancestor_depth]:
                break
            ancestors.append((ancestor_depth, ancestor_idx))
            ancestor_idx //= self.d
        for ancestor_depth, ancestor_idx in reversed(ancestors):
            self._sample(ancestor_depth, ancestor_idx)
        return self._sample(depth, idx)

    def get_subtree_or_sample(self, depth, idx, keep_memory=False) -> BroadcastTree:
        assert 0 <= idx < self.d ** depth
        if idx in self.sampled_subtrees[depth]:
            subtree = self.sampled_subtrees[depth][idx]
            if subtree is None:
                raise ValueError("subtree was sampled but was not kept in memory")
            return subtree
        if self.is_sampled(depth, idx):
            raise ValueError("value or subtree already sampled in this position")
        rho_flip = self.rho * self.get_value_or_sample(depth - 1, idx // self.d) if depth > 0 else 0
        subtree_root_prob = [(1 - rho_flip) / 2, (1 + rho_flip) / 2]
        subtree_conf = BroadcastConfig(self.d, self.rho, self.height - depth)
        subtree = BroadcastTree(subtree_conf, root_prob=subtree_root_prob, seed=self.rng)
        subtree.sample()
        self.sampled_values[depth][idx] = subtree.root
        self.sampled_subtrees[depth][idx] = subtree if keep_memory else None
        return subtree

    def sample_subtree_stream(self, depth, idx, batch_height):
        assert 0 <= idx < self.d ** depth
        if self.is_sampled(depth, idx):
            raise ValueError("subtree or value already sampled")
        self.sampled_subtrees[depth][idx] = None
        rho_flip = self.rho * self.get_value_or_sample(depth - 1, idx // self.d) if depth > 0 else 0
        spin_prob = [(1 - rho_flip) / 2, (1 + rho_flip) / 2]
        target_height = self.height - depth
        batch_height = min(target_height, batch_height)
        num_batches = self.d ** (target_height - batch_height)
        ancestors = []
        for i in range(target_height - batch_height):
            spin = self.rng.choice([-1, 1], p=spin_prob)
            ancestors.append(spin)
            rho_flip = self.rho * spin
            spin_prob = [(1 - rho_flip) / 2, (1 + rho_flip) / 2]
        for tree_pos in range(num_batches):
            if tree_pos > 0:
                zero_cnt = d_order(tree_pos, self.d)
                for i in range(target_height - batch_height - zero_cnt, target_height - batch_height):
                    rho_flip = self.rho * ancestors[i - 1]
                    spin_prob = [(1 - rho_flip) / 2, (1 + rho_flip) / 2]
                    ancestors[i] = self.rng.choice([-1, 1], p=spin_prob)
                rho_flip = self.rho * ancestors[-1]
                spin_prob = [(1 - rho_flip) / 2, (1 + rho_flip) / 2]
            batch_config = BroadcastConfig(self.d, self.rho, batch_height)
            batch_tree = BroadcastTree(batch_config, root_prob=spin_prob, seed=self.rng)
            batch_tree.sample()
            self.sampled_values[depth][idx] = ancestors[0] if ancestors else batch_tree.root
            yield batch_tree, ancestors

    def segment_stream(self, start, end):
        assert 0 <= start <= end <= self.num_leaves
        idx = start
        while idx < end:
            segment_depth = self.height
            segment_len = 1
            segment_idx = idx
            while segment_idx % self.d == 0 and idx + segment_len * self.d <= end:
                segment_depth -= 1
                segment_len *= self.d
                segment_idx //= self.d
            yield segment_depth, segment_idx
            idx += segment_len

    def sample_segment_stream(self, start, end, batch_height):
        for depth, idx in self.segment_stream(start, end):
            ancestors = []
            ancestor_idx = idx // self.d
            for i in range(depth):
                ancestor_depth = depth - 1 - i
                ancestors.append(self.get_value_or_sample(ancestor_depth, ancestor_idx))
                ancestor_idx //= self.d
            ancestors.reverse()
            for batch_tree, subtree_ancestors in self.sample_subtree_stream(depth, idx, batch_height):
                yield batch_tree, ancestors + subtree_ancestors


def broadcast_factory(config: BroadcastConfig, num_trees=1, root_prob=None, seed=None):
    if num_trees == 1:
        return BroadcastTree(config, root_prob=root_prob, seed=seed)
    return BroadcastForest(config, num_trees=num_trees, root_prob=root_prob, seed=seed)


def dynamic_broadcast_tree(config: BroadcastConfig, batch_height, num_trees=1, seed=None):
    ancestors = []
    leaf_idx = 0
    d, rho, height = config.d, config.rho, config.height
    batch_height = min(height, batch_height)
    batch_len = d**batch_height
    sibling_indices = []
    rng = np.random.default_rng(seed)
    flip_prob = [(1 - rho) / 2, (1 + rho) / 2]
    while len(ancestors) != height - batch_height + 1:
        rho_flip = ancestors[-1] * rho if ancestors else np.zeros(num_trees)
        root_prob = [(1 - rho_flip) / 2, (1 + rho_flip) / 2]
        batch_conf = BroadcastConfig(d, rho, batch_height)
        tree = broadcast_factory(batch_conf, num_trees=num_trees, root_prob=root_prob, seed=rng)
        tree.sample()
        yield leaf_idx, tree, ancestors.copy()
        target_idx = len(ancestors) - 1
        while target_idx >= 0 and sibling_indices[target_idx] == d - 1:
            sibling_indices[target_idx] = 0
            target_idx -= 1
        if target_idx == -1:
            new_root = (ancestors[0] if ancestors else tree.roots) * rng.choice([-1, 1], size=num_trees, p=flip_prob)
            ancestors = [new_root] + ancestors
            sibling_indices = [0] + sibling_indices
            target_idx = 0
        sibling_indices[target_idx] += 1
        target_idx += 1
        while target_idx < len(ancestors):
            ancestors[target_idx] = ancestors[target_idx - 1] * rng.choice([-1, 1], size=num_trees, p=flip_prob)
            target_idx += 1
        leaf_idx += batch_len


def block_autoregressive_tree(config: BroadcastConfig, batch_height, num_trees=1, seed=None):
    d, rho, height = config.d, config.rho, config.height
    rng = np.random.default_rng(seed)
    batch_depth = height - batch_height
    tree = None
    for tree_idx in range(d ** batch_depth):
        if tree_idx == 0:
            rho_flip = np.zeros(num_trees)
        else:
            lca_height = d_order(tree_idx, d) + 1
            rho_flip = tree.roots * (rho ** (2 * lca_height))
        root_prob = [(1 - rho_flip) / 2, (1 + rho_flip) / 2]
        batch_conf = BroadcastConfig(d, rho, batch_height)
        tree = broadcast_factory(batch_conf, num_trees=num_trees, root_prob=root_prob, seed=rng)
        tree.sample()
        yield tree
