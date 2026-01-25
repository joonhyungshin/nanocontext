from nanocontext.utils import d_order, d_divide, get_numpy_rng

from .tree import OrderedTree, BroadcastTree, LazyBroadcastTree, dynamic_broadcast_tree


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
            elif self.is_punc_token(token):
                jump_height = self.subtree_height(token)
                for i in range(jump_height):
                    current_node, created = current_node.get_parent_or_create()
                    if created:
                        current_tree.root = current_node
                for i in range(jump_height):
                    current_node = current_node.create_child()
        yield current_tree


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


def num_tokens_expected(tree, prepend_bos=False):
    res = tree.d ** (tree.height - 1) * (tree.d + 1)
    if not prepend_bos:
        res -= 1
    return res


def tokenized_broadcast_trees_with_summaries(d, rho, height, batch_height, tokenizer, summary_every, seed=None):
    rng = get_numpy_rng(seed, local=True)
    tokens_window = []
    beginning = True
    num_trees = 0
    while True:
        tree = LazyBroadcastTree(d, rho, height, seed=rng)
        num_tokens = num_tokens_expected(tree, prepend_bos=not beginning)
        if beginning:
            summary_indices = range(0, num_tokens, summary_every)
        else:
            start_idx = summary_every - len(tokens_window)
            summary_indices = range(start_idx, num_tokens, summary_every)
        for tokens, summary in tokenizer.tokenize_with_summary_stream(tree, batch_height, summary_indices,
                                                                      prepend_bos=not beginning):
            tokens_window.extend(tokens)
            if len(tokens_window) % summary_every == 0:
                yield num_trees, tokens_window, summary
                tokens_window = []
        beginning = False
        num_trees += 1
