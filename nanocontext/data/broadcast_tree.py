import numpy as np

from nanocontext.utils import d_order, d_divide, uniform_slices_from_concatenation, get_numpy_rng

from .common import tokens_to_data


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


class BroadcastForest:
    def __init__(self, d, rho, height, num_trees=1, root_prob=None, seed=None):
        self.d = d
        self.rho = rho
        self.height = height
        self.root_prob = root_prob if root_prob is not None else [0.5, 0.5]
        self.num_trees = num_trees
        self.values = []
        self.rng = np.random.default_rng(seed)

    def sample(self):
        self.values = [self.rng.choice([-1, 1], size=(self.num_trees, 1), p=self.root_prob)]
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
    def __init__(self, d, rho, height, root_prob=None, seed=None):
        super().__init__(d, rho, height, num_trees=1, root_prob=root_prob, seed=seed)

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


class LazyBroadcastTree:
    def __init__(self, d, rho, height, root_prob=None, seed=None):
        self.d = d
        self.rho = rho
        self.height = height
        self.root_prob = root_prob if root_prob is not None else [0.5, 0.5]
        self.rng = np.random.default_rng(seed)
        self.sampled_values = [{} for _ in range(height + 1)]
        self.sampled_subtrees = [{} for _ in range(height + 1)]

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
        subtree = BroadcastTree(self.d, self.rho, self.height - depth, root_prob=subtree_root_prob, seed=self.rng)
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
            batch_tree = BroadcastTree(self.d, self.rho, batch_height, root_prob=spin_prob, seed=self.rng)
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


def num_tokens_expected(tree, prepend_bos=False):
    res = tree.d ** (tree.height - 1) * (tree.d + 1)
    if not prepend_bos:
        res -= 1
    return res


class SpinTreeTokenizer:
    def __init__(self, max_vocab_size, bos_token=0, neg_token=1, pos_token=2, punc_base_token=8,
                 summary_start_token=3, summary_end_token=4, summary_neg_token=5, summary_pos_token=6,
                 summary_pad_token=7):
        self.max_vocab_size = max_vocab_size
        self.bos_token = bos_token
        self.neg_token = neg_token
        self.pos_token = pos_token
        self.punc_base_token = punc_base_token
        self.summary_start_token = summary_start_token
        self.summary_end_token = summary_end_token
        self.summary_neg_token = summary_neg_token
        self.summary_pos_token = summary_pos_token
        self.summary_pad_token = summary_pad_token

    def punctuation(self, subtree_height, summary=False):
        token = min(subtree_height * 2 + self.punc_base_token, self.max_vocab_size - 2)
        if summary:
            token += 1
        return token

    def is_punc_token(self, token):
        return token >= self.punc_base_token

    def subtree_height(self, punc_token):
        return (punc_token - self.punc_base_token) // 2

    def spin_token(self, spin, summary=False):
        pos_token = self.summary_pos_token if summary else self.pos_token
        neg_token = self.summary_neg_token if summary else self.neg_token
        return neg_token if spin < 0 else pos_token

    def sign(self, spin_token):
        return -1 if spin_token in [self.neg_token, self.summary_neg_token] else 1

    def tokenize_stream(self, tree: OrderedTree | BroadcastTree, prepend_bos=False):
        if prepend_bos:
            yield self.bos_token
        for idx, spin in enumerate(tree.get_leaves()):
            if idx > 0:
                zero_cnt = d_order(idx, tree.d)
                if zero_cnt > 0:
                    yield self.punctuation(zero_cnt)
            yield self.spin_token(spin)

    def tokenize(self, tree: OrderedTree | BroadcastTree, prepend_bos=False):
        return list(self.tokenize_stream(tree, prepend_bos))

    def tokenize_summary_stream(self, summary, pad_to=0, wrap=True):
        if wrap:
            yield self.summary_start_token
        for summary_height, summary_spins in summary:
            yield self.punctuation(summary_height, summary=True)
            for spin in summary_spins:
                yield self.spin_token(spin, summary=True)
            for _ in range(pad_to - len(summary_spins)):
                yield self.summary_pad_token
        if wrap:
            yield self.summary_end_token

    def tokenize_summary(self, summary, pad_to=0):
        return list(self.tokenize_summary_stream(summary, pad_to))

    def empty_summary_tokens(self, height, pad_to=0):
        summary = [(height - 1 - i, []) for i in range(height)]
        return self.tokenize_summary(summary, pad_to=pad_to)

    def tokenize_lazy_stream(self, tree: LazyBroadcastTree, batch_height, prepend_bos=False):
        if prepend_bos:
            yield self.bos_token, None, None
        leaf_idx = 0
        for subtree, ancestors in tree.sample_subtree_stream(0, 0, batch_height):
            if leaf_idx > 0:
                zero_cnt = d_order(leaf_idx, tree.d)
                if zero_cnt > 0:
                    yield self.punctuation(zero_cnt), subtree, ancestors
            for token in self.tokenize_stream(subtree, prepend_bos=False):
                yield token, subtree, ancestors
            leaf_idx += subtree.num_leaves

    def tokenize_with_summary_stream(self, tree: LazyBroadcastTree, batch_height, summary_indices, prepend_bos=False):
        summary = [(tree.height - 1 - i, []) for i in range(tree.height)]
        tokens = []
        token_idx = 0
        cur_subtree = None
        cur_subtree_leaf_idx = 0
        leaf_idx = 0
        last_ancestors = []
        for token, subtree, ancestors in self.tokenize_lazy_stream(tree, batch_height, prepend_bos=prepend_bos):
            if subtree is not cur_subtree:
                cur_subtree = subtree
                cur_subtree_leaf_idx = 0
            if token_idx in summary_indices:
                yield tokens, self.tokenize_summary(summary, pad_to=tree.d)
                tokens.clear()
            tokens.append(token)
            if token == self.bos_token:
                pass
            elif token in [self.pos_token, self.neg_token]:
                summary[-1][1].append(self.sign(token))
                cur_subtree_leaf_idx += 1
                leaf_idx += 1
            elif self.is_punc_token(token):
                punc_height = self.subtree_height(token)
                for i in range(punc_height):
                    summary[tree.height - 1 - i][1].clear()
                if cur_subtree_leaf_idx == 0:
                    height = d_order(leaf_idx, tree.d)
                    assert height == punc_height
                    summary_spin = last_ancestors[tree.height - height - 1]
                else:
                    idx, height = d_divide(cur_subtree_leaf_idx, tree.d)
                    assert height == punc_height
                    summary_spin = subtree.values_at(subtree.height - height)[idx - 1]
                summary[tree.height - 1 - punc_height][1].append(summary_spin)
            else:
                assert False
            token_idx += 1
            last_ancestors = ancestors
        yield tokens, self.tokenize_summary(summary, pad_to=tree.d)

    def decode_trees(self, tokens):
        # Make bos_token in the beginning optional... (due to summary)
        current_tree = OrderedTree()
        current_node = current_tree.root
        summary_context = False
        for token in tokens:
            if token == self.summary_end_token:
                summary_context = False
            elif summary_context:
                continue
            elif token == self.summary_start_token:
                summary_context = True
            elif token == self.bos_token:
                if not current_tree.is_singleton():
                    yield current_tree
                current_tree = OrderedTree()
                current_node = current_tree.root
            elif token in [self.neg_token, self.pos_token]:
                spin = -1 if token == self.neg_token else 1
                current_node.create_child(spin)
            elif token >= self.punc_base_token:
                jump_height = self.subtree_height(token)
                for i in range(jump_height):
                    current_node, created = current_node.get_parent_or_create()
                    if created:
                        current_tree.root = current_node
                for i in range(jump_height):
                    current_node = current_node.create_child()
        yield current_tree


def dynamic_broadcast_tree(d, rho, height, batch_height, seed=None):
    ancestors = []
    leaf_idx = 0
    batch_height = min(height, batch_height)
    batch_len = d**batch_height
    sibling_indices = []
    rng = np.random.default_rng(seed)
    flip_prob = [(1 - rho) / 2, (1 + rho) / 2]
    while len(ancestors) != height - batch_height + 1:
        rho_flip = ancestors[-1] * rho if ancestors else 0
        root_prob = [(1 - rho_flip) / 2, (1 + rho_flip) / 2]
        tree = BroadcastTree(d, rho, batch_height, root_prob=root_prob, seed=rng)
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


def tokenized_broadcast_trees(d, rho, height, batch_height, tokenizer, seed=None):
    rng = get_numpy_rng(seed, local=True)
    while True:
        tree = dynamic_broadcast_tree(d, rho, height, batch_height, seed=rng)
        for leaf_idx, subtree, ancestors in tree:
            tokens = []
            if leaf_idx == 0:
                tokens.append(tokenizer.bos_token)
            else:
                zero_cnt = d_order(leaf_idx, d)
                tokens.append(tokenizer.punctuation(zero_cnt))
            tokens.extend(tokenizer.tokenize(subtree))
            yield tokens


def tokenized_broadcast_trees_with_summaries(d, rho, height, batch_height, tokenizer, summary_every, seed=None):
    rng = get_numpy_rng(seed, local=True)
    tokens_window = []
    beginning = True
    while True:
        tree = LazyBroadcastTree(d, rho, height, seed=rng)
        num_tokens = num_tokens_expected(tree, prepend_bos=True)
        if beginning:
            summary_indices = range(0, num_tokens, summary_every)
            beginning = False
        else:
            start_idx = summary_every - len(tokens_window)
            summary_indices = range(start_idx, num_tokens, summary_every)
        for tokens, summary in tokenizer.tokenize_with_summary_stream(tree, batch_height, summary_indices,
                                                                      prepend_bos=True):
            tokens_window.extend(tokens)
            if len(tokens_window) % summary_every == 0:
                yield tokens_window, summary
                tokens_window = []


def broadcast_tree_stream_data_loader(d, rho, height, batch_size, seq_len, batch_height, tokenizer,
                                      summary=False, device="cpu", seed=None):
    rng = get_numpy_rng(seed, local=True)
    if summary:
        needed_tokens = batch_size * (seq_len + 1)
        summary_len = (d + 1) * height + 2
        content_len = seq_len + 1 - 2 * summary_len
        if content_len <= 0:
            raise ValueError("context size too small")
        stream = tokenized_broadcast_trees_with_summaries(d, rho, height, batch_size, tokenizer, content_len, seed=rng)
        _, all_tokens = next(stream)
        for tokens, summary_write in stream:
            all_tokens += tokens + summary_write
            if len(all_tokens) == needed_tokens:
                x, y = tokens_to_data(all_tokens, batch_size, seq_len, device, compact=False)
                # y[:, :summary_len - 1] = -1
                yield x, y
                all_tokens = []
            all_tokens += summary_write
    else:
        needed_tokens = batch_size * seq_len + 1
        trees = tokenized_broadcast_trees(d, rho, height, batch_height, tokenizer, rng)
        for tokens in uniform_slices_from_concatenation(trees, needed_tokens):
            yield tokens_to_data(tokens, batch_size, seq_len, device, compact=True)


def broadcast_tree_sample_data_loader(d, rho, height, batch_size, seq_len, batch_height, tokenizer,
                                      summary=False, device="cpu", seed=None):
    raise NotImplementedError


def broadcast_tree_data_loader(d, rho, height, batch_size, seq_len, batch_height, tokenizer,
                               mode="stream", summary=False, device="cpu", seed=None):
    if mode == "stream":
        yield from broadcast_tree_stream_data_loader(d, rho, height, batch_size, seq_len, batch_height, tokenizer,
                                                     summary=summary, device=device, seed=seed)
    else:
        yield from broadcast_tree_sample_data_loader(d, rho, height, batch_size, seq_len, batch_height, tokenizer,
                                                     summary=summary, device=device, seed=seed)
