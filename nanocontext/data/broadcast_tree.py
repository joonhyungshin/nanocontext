import numpy as np

from nanocontext.utils import d_order, uniform_slices_from_concatenation, get_numpy_rng

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
        return self.values_at(0)

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


class SpinTreeTokenizer:
    def __init__(self, max_vocab_size, bos_token=0, neg_token=1, pos_token=2, punc_base_token=5,
                 summary_start_token=3, summary_end_token=4):
        self.max_vocab_size = max_vocab_size
        self.bos_token = bos_token
        self.neg_token = neg_token
        self.pos_token = pos_token
        self.punc_base_token = punc_base_token
        self.summary_start_token = summary_start_token
        self.summary_end_token = summary_end_token

    def punctuation(self, subtree_height):
        return min(subtree_height + self.punc_base_token, self.max_vocab_size - 1)

    def spin_token(self, spin):
        return self.neg_token if spin < 0 else self.pos_token

    def tokenize(self, tree, prepend_bos=False):
        tokens = []
        if prepend_bos:
            tokens.append(self.bos_token)
        for idx, spin in enumerate(tree.get_leaves()):
            if idx > 0:
                zero_cnt = d_order(idx, tree.d)
                if zero_cnt > 0:
                    tokens.append(self.punctuation(zero_cnt))
            tokens.append(self.spin_token(spin))
        return tokens

    def tokenize_with_summary(self, tree: BroadcastTree,
                              summary_indices, summary_prepend=None, prepend_bos=False):
        tokens = []
        if prepend_bos:
            tokens.append(self.bos_token)
        for idx, spin in enumerate(tree.get_leaves()):
            if idx in summary_indices:
                tokens.append(self.summary_start_token)
                if summary_prepend:
                    tokens.extend(summary_prepend)
                for summary_depth, summary_spin in tree.summarize(0, idx):
                    tokens.append(self.punctuation(tree.height - summary_depth))
                    tokens.append(self.spin_token(summary_spin))
                tokens.append(self.summary_end_token)
            if idx > 0:
                zero_cnt = d_order(idx, tree.d)
                if zero_cnt > 0:
                    tokens.append(self.punctuation(zero_cnt))
            tokens.append(self.spin_token(spin))
        return tokens

    def decode_trees(self, tokens):
        current_tree = None
        current_node = None
        summary_context = False
        for token in tokens:
            if token == self.summary_end_token:
                summary_context = False
            elif summary_context:
                continue
            elif token == self.summary_start_token:
                summary_context = True
            elif token == self.bos_token:
                if current_tree is not None:
                    yield current_tree
                current_tree = OrderedTree()
                current_node = current_tree.root
            elif token in [self.neg_token, self.pos_token]:
                spin = -1 if token == self.neg_token else 1
                if current_tree is None:
                    raise ValueError(f"invalid tokens: expected {self.bos_token} in the beginning")
                current_node.create_child(spin)
            elif token >= self.punc_base_token:
                jump_height = token - self.punc_base_token
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
    batch_len = d ** batch_height
    while True:
        tree = dynamic_broadcast_tree(d, rho, height, batch_height, seed=rng)
        tree_idx = 0
        summary = []
        for leaf_idx, subtree, ancestors in tree:
            tokens = []
            if leaf_idx == 0:
                tokens.append(tokenizer.bos_token)
            else:
                zero_cnt = d_order(leaf_idx, d)
                tokens.append(tokenizer.punctuation(zero_cnt))
            summary_start_idx = (max(leaf_idx, 1) + summary_every - 1) // summary_every * summary_every - leaf_idx
            summary_indices = range(summary_start_idx, batch_len, summary_every)
            tokens.extend(tokenizer.tokenize_with_summary(subtree, summary_indices, summary_prepend=summary))
            yield tokens
            pop_cnt = d_order(tree_idx + 1, d)
            if pop_cnt > 0:
                summary = summary[:-pop_cnt * (2 * d - 2)]
                summary.append(tokenizer.punctuation(batch_height + pop_cnt))
                summary.append(ancestors[-pop_cnt])
            else:
                summary.append(tokenizer.punctuation(batch_height))
                summary.append(tokenizer.spin_token(subtree.root))
            tree_idx += 1


def broadcast_tree_data_loader(d, rho, height, batch_size, seq_len, batch_height, tokenizer,
                               summary_every=-1, device="cpu", seed=None):
    needed_tokens = batch_size * seq_len + 1
    rng = get_numpy_rng(seed, local=True)
    if summary_every == -1:
        trees = tokenized_broadcast_trees(d, rho, height, batch_height, tokenizer, rng)
    else:
        trees = tokenized_broadcast_trees_with_summaries(d, rho, height, batch_height, tokenizer, summary_every, rng)
    for tokens in uniform_slices_from_concatenation(trees, needed_tokens):
        yield tokens_to_data(tokens, batch_size, seq_len, device)
