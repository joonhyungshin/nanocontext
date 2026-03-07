from nanocontext.data.common import BaseTokenizer
from nanocontext.utils import d_order, d_divide
from nanocontext.tree import (
    AbstractPerfectTree, LazyBroadcastTree, BroadcastPolicy, LinkedOrderedTree, PerfectTreeConfig, ValueDomain
)

from .tree import block_autoregressive_tree


class PerfectTreeTokenizer(BaseTokenizer):
    PUNC_TOKEN_NAME = "punc"
    VAL_TOKEN_NAME = "val"

    variable_tokens = [PUNC_TOKEN_NAME, VAL_TOKEN_NAME]

    def __init__(self, max_vocab_size, domain: ValueDomain):
        super().__init__(max_vocab_size)
        self.domain = domain

    def punctuation(self, jump_height):
        return self.get_variable_token(self.PUNC_TOKEN_NAME, jump_height)

    def jump_height(self, token):
        token_name, shift = self.decode_variable_token(token)
        if token_name == self.PUNC_TOKEN_NAME:
            return shift
        return None

    def tokenize_value(self, value):
        return self.get_variable_token(self.VAL_TOKEN_NAME, self.domain.value_to_index(value))

    def decode_value_token(self, token):
        token_name, shift = self.decode_variable_token(token)
        if token_name == self.VAL_TOKEN_NAME:
            return self.domain.index_to_value(shift)
        return None

    def _get_punc_token(self, tree, leaf_idx):
        if leaf_idx == 0:
            return self.bos_token
        zero_cnt = d_order(leaf_idx, tree.d)
        if zero_cnt > 0:
            return self.punctuation(zero_cnt)
        return None

    def tokenize_stream(self, tree: AbstractPerfectTree, token_start_idx=0, prepend_bos=False):
        if not prepend_bos:
            token_start_idx += 1
        parent_idx, child_idx = divmod(token_start_idx, tree.d + 1)
        check_punc = (child_idx == 0)
        leaf_start_idx = parent_idx * tree.d + max(0, child_idx - 1)
        for idx, value in enumerate(tree.leaves_values_stream(start_idx=leaf_start_idx)):
            leaf_idx = idx + leaf_start_idx
            if check_punc:
                punc_token = self._get_punc_token(tree, leaf_idx)
                if punc_token is not None:
                    yield punc_token
            yield self.tokenize_value(value)
            check_punc = True

    def tokenize(self, tree: AbstractPerfectTree, token_start_idx=0, prepend_bos=False):
        return list(self.tokenize_stream(tree, token_start_idx=token_start_idx, prepend_bos=prepend_bos))

    def tokenize_lazy_stream(self, tree: LazyBroadcastTree, batch_height=None, token_start_idx=0, prepend_bos=False):
        if not prepend_bos:
            token_start_idx += 1
        parent_idx, child_idx = divmod(token_start_idx, tree.d + 1)
        check_punc = (child_idx == 0)
        leaf_idx = parent_idx * tree.d + max(0, child_idx - 1)
        for depth, idx in tree.segment_stream(leaf_idx, tree.num_leaves):
            top_ancestors = tree.get_ancestors_values((depth, idx))
            for subtree, ancestors in tree.subtree_stream(depth, idx, batch_height=batch_height):
                if check_punc:
                    punc_token = self._get_punc_token(tree, leaf_idx)
                    if punc_token is not None:
                        yield punc_token, subtree, (top_ancestors + ancestors).copy()
                for token in self.tokenize_stream(subtree, prepend_bos=False):
                    yield token, subtree, (top_ancestors + ancestors).copy()
                leaf_idx += subtree.num_leaves
                check_punc = True

    def tokenize_lazy(self, tree: LazyBroadcastTree, batch_height=None, token_start_idx=0, prepend_bos=False):
        return [token_data[0] for token_data in self.tokenize_lazy_stream(tree, batch_height=batch_height,
                                                                          token_start_idx=token_start_idx,
                                                                          prepend_bos=prepend_bos)]

    def decode_trees_stream(self, tokens):
        # Make bos_token in the beginning optional... (due to summary)
        current_tree = LinkedOrderedTree(domain=self.domain)
        current_node = current_tree.root
        for token in tokens:
            if token == self.bos_token:
                if not current_tree.is_singleton():
                    yield current_tree
                current_tree = LinkedOrderedTree(domain=self.domain)
                current_node = current_tree.root
            else:
                token_name, shift = self.decode_variable_token(token)
                if token_name == self.VAL_TOKEN_NAME:
                    current_node.create_child(self.domain.index_to_value(shift))
                elif token_name == self.PUNC_TOKEN_NAME:
                    for i in range(shift):
                        current_node, created = current_node.get_parent_or_create()
                        if created:
                            current_tree.root = current_node
                    for i in range(shift):
                        current_node = current_node.create_child()
        yield current_tree

    def decode_trees(self, tokens):
        return list(self.decode_trees_stream(tokens))

    def num_tokens(self, d, height, prepend_bos=False):
        return d ** height + d ** (height - 1) + (1 if prepend_bos else 0)


class BroadcastTreeTokenizer(PerfectTreeTokenizer):
    def __init__(self, max_vocab_size, policy: BroadcastPolicy):
        super().__init__(max_vocab_size, policy.get_domain())
        self.policy = policy

    def tokenized_trees_stream(self, config: PerfectTreeConfig, token_start_idx=0, batch_height=None):
        beginning = True
        while True:
            tree = LazyBroadcastTree(config, self.policy)
            token_stream = self.tokenize_lazy_stream(tree, token_start_idx=token_start_idx,
                                                     batch_height=batch_height, prepend_bos=True)
            for token, _, __ in token_stream:
                yield token
            if beginning:
                beginning = False
                token_start_idx = 0

    def tokenized_markov_subtrees_stream(self, config: PerfectTreeConfig, batch_height=None):
        while True:
            tree_sequence = block_autoregressive_tree(config, self.policy, batch_height=batch_height)
            leaf_idx = 0
            for tree in tree_sequence:
                tokens = []
                if leaf_idx == 0:
                    tokens.append(self.bos_token)
                else:
                    zero_cnt = d_order(leaf_idx, config.d)
                    tokens.append(self.punctuation(zero_cnt))
                leaf_idx += tree.num_leaves
                tokens.extend(self.tokenize(tree))
                yield tokens


class SummaryTokenizer(BroadcastTreeTokenizer):
    SUMMARY_PUNC_TOKEN_NAME = "summary_punc"
    SUMMARY_VAL_TOKEN_NAME = "summary_val"

    bos_token = 0
    summary_start_token = 1
    summary_end_token = 2
    variable_token_base = 3
    variable_tokens = BroadcastTreeTokenizer.variable_tokens + [
        SUMMARY_PUNC_TOKEN_NAME,
        SUMMARY_VAL_TOKEN_NAME,
    ]

    def punctuation(self, jump_height, summary=False):
        token_name = self.SUMMARY_PUNC_TOKEN_NAME if summary else self.PUNC_TOKEN_NAME
        return self.get_variable_token(token_name, jump_height)

    def tokenize_value(self, value, summary=False):
        token_name = self.SUMMARY_VAL_TOKEN_NAME if summary else self.VAL_TOKEN_NAME
        return self.get_variable_token(token_name, self.domain.value_to_index(value))

    def jump_height(self, token):
        token_name, shift = self.decode_variable_token(token)
        if token_name in [self.PUNC_TOKEN_NAME, self.SUMMARY_PUNC_TOKEN_NAME]:
            return shift
        return None

    def decode_value_token(self, token):
        token_name, shift = self.decode_variable_token(token)
        if token_name in [self.VAL_TOKEN_NAME, self.SUMMARY_VAL_TOKEN_NAME]:
            return self.domain.index_to_value(shift)
        return None

    def tokenize_summary_stream(self, summary, config: PerfectTreeConfig):
        raise NotImplementedError

    def tokenize_summary(self, summary, config: PerfectTreeConfig, wrap=True):
        tokens = list(self.tokenize_summary_stream(summary, config))
        if wrap:
            tokens = [self.summary_start_token] + tokens + [self.summary_end_token]
        return tokens

    def init_summary(self, config: PerfectTreeConfig):
        raise NotImplementedError

    def update_summary(self, summary, token_data, config: PerfectTreeConfig):
        raise NotImplementedError

    def prebuild_summary(self, tree: LazyBroadcastTree, token_start_idx=0, batch_height=None, prepend_bos=False):
        summary = self.init_summary(tree.config)
        if not prepend_bos:
            token_start_idx += 1
        token_stream = self.tokenize_lazy_stream(tree, batch_height=batch_height, prepend_bos=False)
        for _ in range(token_start_idx):
            token_data = next(token_stream)
            summary = self.update_summary(summary, token_data, tree.config)
        return summary

    def tokenize_with_summary_stream(self, tree: LazyBroadcastTree, summary_indices, token_start_idx=0,
                                     batch_height=None, prepend_bos=False):
        summary = self.prebuild_summary(tree, token_start_idx=token_start_idx,
                                        batch_height=batch_height, prepend_bos=prepend_bos)
        tokens = []
        token_stream = self.tokenize_lazy_stream(tree, token_start_idx=token_start_idx,
                                                 batch_height=batch_height, prepend_bos=prepend_bos)
        for idx, token_data in enumerate(token_stream):
            token_idx = token_start_idx + idx
            if token_idx in summary_indices:
                yield tokens.copy(), self.tokenize_summary(summary, tree.config)
                tokens.clear()
            tokens.append(token_data[0])
            summary = self.update_summary(summary, token_data, tree.config)
        yield tokens, self.tokenize_summary(summary, tree.config)

    def init_summary_tokens(self, config: PerfectTreeConfig):
        return self.tokenize_summary(self.init_summary(config), config)

    def tokenized_trees_with_summaries_stream(self, config: PerfectTreeConfig, summary_every,
                                              batch_height=None, token_start_idx=0):
        tokens_window = []
        beginning = True
        num_trees = 0
        num_tokens = (config.d ** (config.height - 1)) * (config.d + 1)
        token_start_idx %= num_tokens
        while True:
            tree = LazyBroadcastTree(config, self.policy)
            if beginning:
                summary_indices = range(token_start_idx, num_tokens, summary_every)
                local_start_idx = token_start_idx
            else:
                new_start_idx = summary_every - len(tokens_window)
                summary_indices = range(new_start_idx, num_tokens, summary_every)
                local_start_idx = 0
            for tokens, summary in self.tokenize_with_summary_stream(tree, summary_indices,
                                                                     token_start_idx=local_start_idx,
                                                                     batch_height=batch_height,
                                                                     prepend_bos=not beginning):
                tokens_window.extend(tokens)
                if len(tokens_window) == 0 or len(tokens_window) >= summary_every:
                    yield num_trees, tokens_window, summary
                    tokens_window = []
            beginning = False
            num_trees += 1


class SegmentSummaryTokenizer(SummaryTokenizer):
    bos_token = 0
    summary_start_token = 1
    summary_end_token = 2
    summary_pad_token = 3
    variable_token_base = 4

    def tokenize_summary_stream(self, summary, config, wrap_to=None):
        summary_data, _ = summary
        wrap_to = wrap_to or config.d
        for summary_height, summary_values in summary_data:
            yield self.punctuation(summary_height, summary=True)
            for value in summary_values:
                yield self.tokenize_value(value, summary=True)
            for _ in range(wrap_to - len(summary_values)):
                yield self.summary_pad_token

    def init_summary(self, config):
        ctx = {
            "cur_subtree": None,
            "cur_subtree_leaf_idx": 0,
            "leaf_idx": 0,
            "last_ancestors": []
        }
        return [(config.height - 1 - i, []) for i in range(config.height)], ctx

    def prebuild_summary(self, tree: LazyBroadcastTree, token_start_idx=0, batch_height=None, prepend_bos=False):
        if not prepend_bos:
            token_start_idx += 1
        parent_idx, child_idx = divmod(token_start_idx, tree.d + 1)
        leaf_idx = parent_idx * tree.d + max(0, child_idx - 1)
        summary_data, ctx = self.init_summary(tree.config)
        if child_idx == 0 and leaf_idx > 0:
            for depth, idx in tree.segment_stream(0, leaf_idx - 1):
                summary_data[depth - 1][1].append(tree.get_value_or_sample(depth, idx))
            summary_data[tree.height - 1][1].append(tree.get_value_or_sample(tree.height - 1, leaf_idx - 1))
        else:
            if leaf_idx == tree.num_leaves:
                return summary_data, ctx
            for depth, idx in tree.segment_stream(0, leaf_idx):
                summary_data[depth - 1][1].append(tree.get_value_or_sample(depth, idx))
        ctx["leaf_idx"] = leaf_idx
        if leaf_idx > 0:
            ctx["last_ancestors"] = tree.get_ancestors_of_leaf(leaf_idx - 1)
        return summary_data, ctx

    def update_summary(self, summary, token_data, config):
        token, subtree, ancestors = token_data
        summary_data, ctx = summary
        if ctx["cur_subtree"] is not subtree:
            ctx["cur_subtree"] = subtree
            ctx["cur_subtree_leaf_idx"] = 0
        if token != self.bos_token:
            token_name, shift = self.decode_variable_token(token)
            if token_name == self.VAL_TOKEN_NAME:
                summary_data[-1][1].append(self.domain.index_to_value(shift))
                ctx["cur_subtree_leaf_idx"] += 1
                ctx["leaf_idx"] += 1
            elif token_name == self.PUNC_TOKEN_NAME:
                for i in range(shift):
                    summary_data[config.height - 1 - i][1].clear()
                if ctx["cur_subtree_leaf_idx"] == 0:
                    height = d_order(ctx["leaf_idx"], config.d)
                    assert height == shift
                    summary_val = ctx["last_ancestors"][config.height - shift]
                else:
                    idx, height = d_divide(ctx["cur_subtree_leaf_idx"], config.d)
                    assert height == shift
                    summary_val = subtree.value_at(subtree.height - height, idx - 1)
                summary_data[config.height - 1 - shift][1].append(summary_val)
            else:
                assert False
        ctx["last_ancestors"] = ancestors + [subtree.get_root_value()]
        return summary_data, ctx


class HierarchySummaryTokenizer(SummaryTokenizer):
    bos_token = 0
    summary_start_token = 1
    summary_end_token = 2
    summary_pad_token = 3
    variable_token_base = 4

    SIBLING_IDX_TOKEN_NAME = "sibling_idx"
    variable_tokens = SummaryTokenizer.variable_tokens + [
        SIBLING_IDX_TOKEN_NAME,
    ]

    def sibling_index_token(self, idx):
        return self.get_variable_token(self.SIBLING_IDX_TOKEN_NAME, idx)

    def tokenize_summary_stream(self, summary, config):
        summary_data, ctx = summary
        for value, node_index in summary_data:
            yield self.summary_pad_token if value is None else self.tokenize_value(value, summary=True)
            yield self.sibling_index_token(node_index)

    def init_summary(self, config: PerfectTreeConfig):
        ctx = {
            "cur_subtree": None,
            "cur_subtree_leaf_idx": 0,
        }
        return [(None, 0) for _ in range(config.height)], ctx

    def prebuild_summary(self, tree: LazyBroadcastTree, token_start_idx=0, batch_height=None, prepend_bos=False):
        if not prepend_bos:
            token_start_idx += 1
        parent_idx, child_idx = divmod(token_start_idx, tree.d + 1)
        leaf_idx = parent_idx * tree.d + max(0, child_idx - 1)
        summary_data, ctx = self.init_summary(tree.config)
        if leaf_idx > 0:
            ancestors = tree.get_ancestors_of_leaf(leaf_idx - 1)
            if child_idx == 0:
                summary_data[-1] = (ancestors[-1], tree.d)
                node_idx = (leaf_idx - 1) // tree.d
                for i in range(1, tree.height):
                    ancestor_hint = ancestors[tree.height - 1 - i]
                    summary_data[tree.height - 1 - i] = (ancestor_hint, node_idx % tree.d)
                    node_idx //= tree.d
            else:
                zero_cnt = d_order(leaf_idx, tree.d)
                node_idx = leaf_idx
                for i in range(tree.height):
                    ancestor_hint = ancestors[tree.height - 1 - i] if i >= zero_cnt else None
                    summary_data[tree.height - 1 - i] = (ancestor_hint, node_idx % tree.d)
                    node_idx //= tree.d
        return summary_data, ctx

    def update_summary(self, summary, token_data, config: PerfectTreeConfig):
        token, subtree, ancestors = token_data
        summary_data, ctx = summary
        if ctx["cur_subtree"] is not subtree:
            ctx["cur_subtree"] = subtree
            ctx["cur_subtree_leaf_idx"] = 0
        if token != self.bos_token:
            token_name, shift = self.decode_variable_token(token)
            if token_name == self.VAL_TOKEN_NAME:
                full_ancestors = ancestors + subtree.get_ancestors_of_leaf(ctx["cur_subtree_leaf_idx"])
                for i in range(config.height):
                    depth = config.height - 1 - i
                    parent_spin, sibling_idx = summary_data[depth]
                    if parent_spin is None:
                        parent_spin = full_ancestors[depth]
                        summary_data[depth] = (parent_spin, sibling_idx)
                    else:
                        break
                ctx["cur_subtree_leaf_idx"] += 1
                parent_spin, sibling_idx = summary_data[-1]
                summary_data[-1] = (parent_spin, sibling_idx + 1)
            elif token_name == self.PUNC_TOKEN_NAME:
                for i in range(shift):
                    summary_data[config.height - 1 - i] = (None, 0)
                spin, sibling_idx = summary_data[config.height - 1 - shift]
                summary_data[config.height - 1 - shift] = (spin, sibling_idx + 1)
            else:
                assert False
        return summary_data, ctx
