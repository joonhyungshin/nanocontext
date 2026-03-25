from nanocontext.data.common import tokens_to_data
from nanocontext.tree import BroadcastPolicy, LazyBroadcastTree
from nanocontext.utils import get_numpy_rng, uniform_slices_from_concatenation, d_order

from .tree import PerfectTreeConfig, block_autoregressive_tree
from .tokenizer import SummaryTokenizer, PerfectTreeTokenizer


class BroadcastTreeStreamer:
    def __init__(self, tokenizer: PerfectTreeTokenizer, tree_config: PerfectTreeConfig, policy: BroadcastPolicy):
        self.tokenizer = tokenizer
        self.policy = policy
        self.tree_config = tree_config

    def tokenized_trees_stream(self, token_start_idx=0, batch_height=None):
        beginning = True
        while True:
            tree = LazyBroadcastTree(self.tree_config, self.policy)
            token_stream = self.tokenizer.tokenize_lazy_stream(tree, token_start_idx=token_start_idx,
                                                               batch_height=batch_height, prepend_bos=True)
            for token, _, __ in token_stream:
                yield token
            if beginning:
                beginning = False
                token_start_idx = 0

    def tokenized_markov_subtrees_stream(self, batch_height=None):
        while True:
            tree_sequence = block_autoregressive_tree(self.tree_config, self.policy, batch_height=batch_height)
            leaf_idx = 0
            for tree in tree_sequence:
                tokens = []
                if leaf_idx == 0:
                    tokens.append(self.tokenizer.bos_token)
                else:
                    zero_cnt = d_order(leaf_idx, self.tree_config.d)
                    tokens.append(self.tokenizer.punctuation(zero_cnt))
                leaf_idx += tree.num_leaves
                tokens.extend(self.tokenizer.tokenize(tree))
                yield tokens

    def tokenized_trees_with_summaries_stream(self, summary_every,
                                              batch_height=None, token_start_idx=0):
        if not isinstance(self.tokenizer, SummaryTokenizer):
            raise ValueError("tokenizer does not support summarizing")
        tokens_window = []
        beginning = True
        num_trees = 0
        config = self.tree_config
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
            for tokens, summary in self.tokenizer.tokenize_with_summary_stream(tree, summary_indices,
                                                                               token_start_idx=local_start_idx,
                                                                               batch_height=batch_height,
                                                                               prepend_bos=not beginning):
                tokens_window.extend(tokens)
                if len(tokens_window) == 0 or len(tokens_window) >= summary_every:
                    yield num_trees, tokens_window.copy(), summary
                    tokens_window = []
            beginning = False
            num_trees += 1


def broadcast_tree_stream_data_loader(tokenizer: PerfectTreeTokenizer, config: PerfectTreeConfig,
                                      policy: BroadcastPolicy, batch_size, seq_len,
                                      batch_height=None, summary=False, device="cpu"):
    streamer = BroadcastTreeStreamer(tokenizer, config, policy)
    if summary:
        if not isinstance(tokenizer, SummaryTokenizer):
            raise ValueError("tokenizer does not support summarizing")
        needed_tokens = batch_size * (seq_len + 1)
        summary_len = len(tokenizer.init_summary_tokens(config))
        content_len = seq_len + 1 - 2 * summary_len
        if content_len <= 0:
            raise ValueError("context size too small")
        stream = streamer.tokenized_trees_with_summaries_stream(content_len, batch_height=batch_height)
        _, _, all_tokens = next(stream)
        for num_trees, tokens, summary_write in stream:
            all_tokens += tokens + summary_write
            if len(all_tokens) == needed_tokens:
                x, y = tokens_to_data(all_tokens, batch_size, seq_len, device, compact=False)
                y = y.clone()
                y[:, :summary_len - 1] = -1
                yield x, y
                all_tokens = []
            all_tokens += summary_write
    else:
        needed_tokens = batch_size * seq_len + 1
        trees = streamer.tokenized_trees_stream(batch_height=batch_height)
        for tokens in uniform_slices_from_concatenation(trees, needed_tokens):
            yield tokens_to_data(tokens, batch_size, seq_len, device, compact=True)


def broadcast_tree_sample_data_loader(tokenizer: PerfectTreeTokenizer, config: PerfectTreeConfig,
                                      policy: BroadcastPolicy, batch_size, seq_len,
                                      batch_height=None, summary=False, device="cpu", seed=None):
    rng = get_numpy_rng(seed, local=True)
    d, height = config.d, config.height
    num_tokens = (d ** (height - 1)) * (d + 1)
    streamer = BroadcastTreeStreamer(tokenizer, config, policy)
    if summary:
        if not isinstance(tokenizer, SummaryTokenizer):
            raise ValueError("tokenizer does not support summarizing")
        summary_len = len(tokenizer.init_summary_tokens(config))
        content_len = seq_len + 1 - 2 * summary_len
        if content_len <= 0:
            raise ValueError("context size too small")
        while True:
            tokens = []
            for _ in range(batch_size):
                start_idx = rng.integers(0, num_tokens - 1)
                stream = streamer.tokenized_trees_with_summaries_stream(content_len,
                                                                        batch_height=batch_height,
                                                                        token_start_idx=start_idx)
                _, _, all_tokens = next(stream)
                num_trees, cur_tokens, summary_write = next(stream)
                all_tokens += cur_tokens + summary_write
                tokens.extend(all_tokens)
            x, y = tokens_to_data(tokens, batch_size, seq_len, device, compact=False)
            y = y.clone()
            y[:, :summary_len - 1] = -1
            yield x, y
    else:
        while True:
            tokens = []
            for _ in range(batch_size):
                start_idx = rng.integers(0, num_tokens - 1)
                trees = streamer.tokenized_trees_stream(token_start_idx=start_idx, batch_height=batch_height)
                tokens.extend(next(uniform_slices_from_concatenation(trees, seq_len + 1)))
            yield tokens_to_data(tokens, batch_size, seq_len, device, compact=False)


def broadcast_tree_data_loader(tokenizer: PerfectTreeTokenizer, config: PerfectTreeConfig,
                               policy: BroadcastPolicy, batch_size, seq_len,
                               batch_height=None, mode="stream", summary=False, device="cpu", seed=None):
    if mode == "stream":
        yield from broadcast_tree_stream_data_loader(tokenizer, config, policy, batch_size, seq_len,
                                                     batch_height=batch_height, summary=summary, device=device)
    else:
        yield from broadcast_tree_sample_data_loader(tokenizer, config, policy, batch_size, seq_len,
                                                     batch_height=batch_height, summary=summary, device=device,
                                                     seed=seed)


def block_autoregressive_tree_data_loader(tokenizer: PerfectTreeTokenizer, config: PerfectTreeConfig,
                                          policy: BroadcastPolicy, batch_size, seq_len,
                                          batch_height=None, device="cpu"):
    needed_tokens = batch_size * seq_len + 1
    streamer = BroadcastTreeStreamer(tokenizer, config, policy)
    trees = streamer.tokenized_markov_subtrees_stream(batch_height=batch_height)
    for tokens in uniform_slices_from_concatenation(trees, needed_tokens):
        yield tokens_to_data(tokens, batch_size, seq_len, device, compact=True)
