import numpy as np

from . import PerfectSubtree, ValueDomain
from .common import AbstractPerfectTree, PerfectTreeConfig
from ..utils import d_order


class BroadcastPolicy:
    def get_domain(self) -> ValueDomain:
        raise NotImplementedError

    def broadcast(self, values=None, multi=1) -> np.ndarray:
        raise NotImplementedError


class BroadcastForest:
    def __init__(self, config: PerfectTreeConfig, policy: BroadcastPolicy,
                 num_trees=1, root_values=None, seed=None):
        self.config = config
        self.policy = policy
        self.domain = policy.get_domain()
        self.num_trees = num_trees
        self.root_values = root_values
        self.rng = np.random.default_rng(seed=seed)
        self.values = []

    @property
    def d(self):
        return self.config.d

    @property
    def height(self):
        return self.config.height

    def _sample_roots(self):
        if self.root_values is None:
            self.values = [self.policy.broadcast(multi=self.num_trees)[:, np.newaxis]]
        else:
            self.values = [np.broadcast_to(self.root_values, (self.num_trees, 1))]

    def get_root_values(self):
        return self.values[0]

    def sample(self):
        self._sample_roots()
        cur_size = 1
        for i in range(self.height):
            children = self.policy.broadcast(self.values[-1], multi=self.d).swapaxes(0, 1)
            self.values.append(children.reshape((self.num_trees, -1), order='F'))
            cur_size *= self.d

    @property
    def sampled(self):
        return len(self.values) == self.height + 1

    def get_value(self, tree_idx: int, depth: int, idx: int):
        if not self.sampled:
            self.sample()
        return self.values[depth][tree_idx, idx].item()

    def get_tree(self, tree_idx):
        tree = BroadcastTree(config=self.config, policy=self.policy)
        tree._forest = self
        tree._tree_idx = tree_idx
        return tree

    def __getitem__(self, tree_idx):
        return self.get_tree(tree_idx)

    def __str__(self):
        if not self.sampled:
            return "(not sampled)"
        msg = ""
        for j in range(self.num_trees):
            for i, layer in enumerate(self.values):
                msg += (" " * (self.d ** (self.height - i) - 1)).join([self.domain.value_to_char(node)
                                                                       for node in layer[j]])
                msg += "\n"
        return msg


class BroadcastTree(AbstractPerfectTree):
    def __init__(self, config: PerfectTreeConfig, policy: BroadcastPolicy,
                 root_value=None):
        super().__init__(config)
        self.policy = policy
        self.domain = policy.get_domain()
        self.root_value = root_value
        self._forest = None
        self._tree_idx = None

    @property
    def forest(self):
        if self._forest is None:
            self._forest = BroadcastForest(config=self.config, policy=self.policy, root_values=self.root_value)
            self._tree_idx = 0
        return self._forest

    def sample(self):
        self.forest.sample()

    def value_at(self, depth, idx):
        return self.forest.get_value(self._tree_idx, depth, idx)

    def value_to_char(self, value):
        self.domain.value_to_char(value)


class LazyBroadcastTree(AbstractPerfectTree):
    def __init__(self, config: PerfectTreeConfig, policy: BroadcastPolicy):
        super().__init__(config)
        self.policy = policy
        self.sampled_values = [{} for _ in range(self.height + 1)]
        self.sampled_subtrees = [{} for _ in range(self.height + 1)]

    def broadcast_subtree(self, height, root_value=None) -> AbstractPerfectTree:
        subtree_conf = PerfectTreeConfig(d=self.d, height=height)
        subtree = BroadcastTree(subtree_conf, self.policy, root_value=root_value)
        subtree.sample()
        return subtree

    def _sample(self, depth, idx):
        assert 0 <= idx < self.d ** depth
        assert depth == 0 or idx // self.d in self.sampled_values[depth - 1]
        assert idx not in self.sampled_values[depth]
        value = self.get_value_or_sample(depth - 1, idx // self.d) if depth > 0 else None
        self.sampled_values[depth][idx] = self.policy.broadcast(value).item()
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
        ancestors = []
        ancestor_idx = idx // self.d
        for i in range(depth):
            ancestor_depth = depth - 1 - i
            if ancestor_idx in self.sampled_subtrees[ancestor_depth]:
                subtree = self.sampled_subtrees[ancestor_depth][ancestor_idx]
                if subtree is None:
                    raise ValueError("subtree was sampled but was not kept in memory")
                rel_depth = depth - ancestor_depth
                rel_idx = idx - ancestor_idx * (self.d ** rel_depth)
                return subtree.value_at(rel_depth, rel_idx)
            elif ancestor_idx in self.sampled_values[ancestor_depth]:
                break
            ancestors.append((ancestor_depth, ancestor_idx))
            ancestor_idx //= self.d
        for ancestor_depth, ancestor_idx in reversed(ancestors):
            self._sample(ancestor_depth, ancestor_idx)
        return self._sample(depth, idx)

    def value_at(self, depth, idx):
        return self.get_value_or_sample(depth, idx)

    def get_subtree_or_sample(self, depth, idx, keep_memory=False) -> AbstractPerfectTree:
        assert 0 <= idx < self.d ** depth
        if idx in self.sampled_subtrees[depth]:
            subtree = self.sampled_subtrees[depth][idx]
            if subtree is None:
                raise ValueError("subtree was sampled but was not kept in memory")
            return subtree
        elif idx in self.sampled_values[depth]:
            raise ValueError("value already sampled in this position")
        ancestor_idx = idx // self.d
        for i in range(depth):
            ancestor_depth = depth - 1 - i
            if ancestor_idx in self.sampled_subtrees[ancestor_depth]:
                subtree = self.sampled_subtrees[ancestor_depth][ancestor_idx]
                if subtree is None:
                    raise ValueError("subtree was sampled but was not kept in memory")
                rel_depth = depth - ancestor_depth
                rel_idx = idx - ancestor_idx * (self.d ** rel_depth)
                return PerfectSubtree(subtree, rel_depth, rel_idx)
            ancestor_idx //= self.d
        value = self.get_value_or_sample(depth, idx)
        subtree = self.broadcast_subtree(self.height - depth, root_value=value)
        self.sampled_subtrees[depth][idx] = subtree if keep_memory else None
        return subtree

    def subtree_stream(self, depth, idx, batch_height=None):
        assert 0 <= idx < self.d ** depth
        subtree_idx = idx
        for i in range(depth + 1):
            subtree_depth = depth - i
            if subtree_idx in self.sampled_subtrees[subtree_depth]:
                subtree = self.sampled_subtrees[subtree_depth][subtree_idx]
                if subtree is None:
                    raise ValueError("subtree was sampled but was not kept in memory")
                rel_depth = depth - subtree_depth
                rel_idx = idx - subtree_idx * (self.d ** rel_depth)
                yield from subtree.subtree_stream(rel_depth, rel_idx, batch_height=batch_height)
                return
            elif subtree_idx in self.sampled_values[subtree_depth]:
                if subtree_idx == idx:
                    raise ValueError("value was already sampled in this position")
                break
            subtree_idx //= self.d
        self.sampled_subtrees[depth][idx] = None
        cur_value = self.get_value_or_sample(depth - 1, idx // self.d) if depth > 0 else None
        target_height = self.height - depth
        batch_height = min(target_height, batch_height) if batch_height is not None else target_height
        num_batches = self.d ** (target_height - batch_height)
        ancestors = []
        for i in range(target_height - batch_height):
            cur_value = self.policy.broadcast(cur_value).item()
            ancestors.append(cur_value)
        for tree_pos in range(num_batches):
            if tree_pos > 0:
                zero_cnt = d_order(tree_pos, self.d)
                for i in range(target_height - batch_height - zero_cnt, target_height - batch_height):
                    cur_value = ancestors[i - 1]
                    ancestors[i] = self.policy.broadcast(cur_value).item()
                cur_value = ancestors[-1]
            subtree_root_value = self.policy.broadcast(cur_value).item()
            batch_tree = self.broadcast_subtree(batch_height, root_value=subtree_root_value)
            self.sampled_values[depth][idx] = ancestors[0] if ancestors else subtree_root_value
            yield batch_tree, ancestors.copy()

    def sample_segment_stream(self, start, end, batch_height=None):
        for depth, idx in self.segment_stream(start, end):
            ancestors = []
            ancestor_idx = idx // self.d
            for i in range(depth):
                ancestor_depth = depth - 1 - i
                ancestors.append(self.get_value_or_sample(ancestor_depth, ancestor_idx))
                ancestor_idx //= self.d
            ancestors.reverse()
            for batch_tree, subtree_ancestors in self.subtree_stream(depth, idx, batch_height=batch_height):
                yield batch_tree, ancestors + subtree_ancestors
