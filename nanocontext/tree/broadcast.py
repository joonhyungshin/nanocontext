import numpy as np

from . import PerfectSubtree, StateSpace, AbstractOrderedTree
from .common import AbstractPerfectTree, PerfectTreeConfig
from ..utils import d_order


class BroadcastChannel:
    def get_state_space(self) -> StateSpace:
        raise NotImplementedError

    def broadcast(self, values=None, multi=1) -> np.ndarray:
        raise NotImplementedError


class FiniteBroadcastChannel(BroadcastChannel):
    def __init__(self, state_space: StateSpace, kernel: np.ndarray):
        self.state_space = state_space
        self.kernel = kernel

    def get_state_space(self):
        return self.state_space


class BroadcastForest:
    def __init__(self, config: PerfectTreeConfig, channel: BroadcastChannel,
                 num_trees=1, root_values=None):
        self.config = config
        self.channel = channel
        self.value_space = channel.get_state_space()
        self.num_trees = num_trees
        self.root_values = root_values
        self.values = []

    @property
    def d(self):
        return self.config.d

    @property
    def height(self):
        return self.config.height

    def _sample_roots(self):
        if self.root_values is None:
            self.values = [self.channel.broadcast(multi=self.num_trees)[:, np.newaxis]]
        else:
            self.values = [np.broadcast_to(self.root_values, (self.num_trees,))[:, np.newaxis]]

    def get_root_values(self):
        return self.values[0][:, 0]

    def get_leaves(self):
        return self.values[self.height]

    def sample(self):
        self._sample_roots()
        cur_size = 1
        for i in range(self.height):
            children = self.channel.broadcast(self.values[-1], multi=self.d).swapaxes(0, 1)
            self.values.append(children.reshape((self.num_trees, -1), order='F'))
            cur_size *= self.d

    @property
    def sampled(self):
        return len(self.values) == self.height + 1

    def get_values(self, depth: int, idx: int) -> np.ndarray:
        return self.values[depth][:, idx]

    def get_value(self, tree_idx: int, depth: int, idx: int):
        if not self.sampled:
            self.sample()
        return self.values[depth][tree_idx, idx].item()

    def get_tree(self, tree_idx):
        tree = BroadcastTree(config=self.config, channel=self.channel)
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
                msg += (" " * (self.d ** (self.height - i) - 1)).join([self.value_space.state_to_char(node)
                                                                       for node in layer[j]])
                msg += "\n"
        return msg


class BroadcastTree(AbstractPerfectTree):
    def __init__(self, config: PerfectTreeConfig, channel: BroadcastChannel,
                 root_value=None):
        super().__init__(config)
        self.channel = channel
        self.value_space = channel.get_state_space()
        self.root_value = root_value
        self._forest = None
        self._tree_idx = None

    @property
    def forest(self):
        if self._forest is None:
            self._forest = BroadcastForest(config=self.config, channel=self.channel, root_values=self.root_value)
            self._tree_idx = 0
        return self._forest

    def sample(self):
        self.forest.sample()

    def value_at(self, depth, idx):
        return self.forest.get_value(self._tree_idx, depth, idx)

    def value_to_char(self, value):
        self.value_space.state_to_char(value)

    @classmethod
    def from_generic(cls, tree: AbstractOrderedTree):
        pass


class LazyBroadcastTree(AbstractPerfectTree):
    def __init__(self, config: PerfectTreeConfig, channel: BroadcastChannel):
        super().__init__(config)
        self.channel = channel
        self.sampled_values = [{} for _ in range(self.height + 1)]
        self.sampled_subtrees = [{} for _ in range(self.height + 1)]

    def broadcast_subtree(self, height, root_value=None) -> AbstractPerfectTree:
        subtree_conf = PerfectTreeConfig(d=self.d, height=height)
        subtree = BroadcastTree(subtree_conf, self.channel, root_value=root_value)
        subtree.sample()
        return subtree

    def _sample(self, depth, idx):
        assert 0 <= idx < self.d ** depth
        assert depth == 0 or idx // self.d in self.sampled_values[depth - 1]
        assert idx not in self.sampled_values[depth]
        value = self.get_value_or_sample(depth - 1, idx // self.d) if depth > 0 else None
        self.sampled_values[depth][idx] = self.channel.broadcast(value).item()
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
            cur_value = self.channel.broadcast(cur_value).item()
            ancestors.append(cur_value)
        for tree_pos in range(num_batches):
            if tree_pos > 0:
                zero_cnt = d_order(tree_pos, self.d)
                for i in range(target_height - batch_height - zero_cnt, target_height - batch_height):
                    cur_value = ancestors[i - 1]
                    ancestors[i] = self.channel.broadcast(cur_value).item()
                cur_value = ancestors[-1]
            subtree_root_value = self.channel.broadcast(cur_value).item()
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


class InferenceTree(AbstractPerfectTree):
    def __init__(self, config, leaves):
        super().__init__(config)
        self.leaves = leaves

    def value_at(self, depth, idx):
        return self.leaves[idx] if depth == self.height else None


def markov_forest(config: PerfectTreeConfig, channel: BroadcastChannel, batch_height=None, num_trees=1, seed=None):
    rng = np.random.default_rng(seed)
    d, height = config.d, config.height
    batch_height = min(height, batch_height) if batch_height is not None else height
    batch_depth = height - batch_height
    forest = None
    probs = [1]
    for i in range(batch_depth):
        probs.append(config.d - 1 if i == 0 else probs[-1] * config.d)
    for i in range(batch_depth + 1):
        probs[i] /= config.d ** batch_depth
    for tree_idx in range(d ** batch_depth):
        if tree_idx == 0:
            root_values = None
        else:
            root_values = forest.get_root_values().copy()
            membership = rng.choice(range(batch_depth, -1, -1), size=num_trees, p=probs)
            for i in range(batch_depth):
                target_idx = (membership >= i)
                for _ in range(2):
                    root_values[target_idx] = channel.broadcast(root_values[target_idx]).squeeze(0)
            reset_idx = (membership == batch_depth)
            root_values[reset_idx] = channel.broadcast(multi=np.sum(reset_idx))
        batch_conf = PerfectTreeConfig(d, batch_height)
        forest = BroadcastForest(batch_conf, channel, root_values=root_values, num_trees=num_trees)
        forest.sample()
        yield forest


def markov_tree(config: PerfectTreeConfig, channel: BroadcastChannel, batch_height=None, seed=None):
    for forest in markov_forest(config, channel, batch_height=batch_height, seed=seed):
        yield forest[0]
