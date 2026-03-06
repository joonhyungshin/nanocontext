from nanocontext.data.common import tokens_to_data
from nanocontext.utils import get_numpy_rng, uniform_slices_from_concatenation

from .tree import PerfectTreeConfig
from .tokenizer import SummaryTokenizer, BroadcastTreeTokenizer


def broadcast_tree_stream_data_loader(tokenizer: BroadcastTreeTokenizer, config: PerfectTreeConfig, batch_size, seq_len,
                                      batch_height=None, summary=False, device="cpu"):
    if summary:
        if not isinstance(tokenizer, SummaryTokenizer):
            raise ValueError("tokenizer does not support summarizing")
        needed_tokens = batch_size * (seq_len + 1)
        summary_len = len(tokenizer.init_summary_tokens(config))
        content_len = seq_len + 1 - 2 * summary_len
        if content_len <= 0:
            raise ValueError("context size too small")
        stream = tokenizer.tokenized_trees_with_summaries_stream(config, content_len, batch_height=batch_height)
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
        trees = tokenizer.tokenized_trees_stream(config, batch_height=batch_height)
        for tokens in uniform_slices_from_concatenation(trees, needed_tokens):
            yield tokens_to_data(tokens, batch_size, seq_len, device, compact=True)


def broadcast_tree_sample_data_loader(tokenizer: BroadcastTreeTokenizer, config: PerfectTreeConfig, batch_size, seq_len,
                                      batch_height=None, summary=False, device="cpu", seed=None):
    rng = get_numpy_rng(seed, local=True)
    num_tokens = tokenizer.num_tokens(config.d, config.height, prepend_bos=True)
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
                stream = tokenizer.tokenized_trees_with_summaries_stream(config, content_len,
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
                trees = tokenizer.tokenized_trees_stream(config, token_start_idx=start_idx, batch_height=batch_height)
                tokens.extend(next(uniform_slices_from_concatenation(trees, seq_len + 1)))
            yield tokens_to_data(tokens, batch_size, seq_len, device, compact=False)


def broadcast_tree_data_loader(tokenizer: BroadcastTreeTokenizer, config: PerfectTreeConfig, batch_size, seq_len,
                               batch_height=None, mode="stream", summary=False, device="cpu", seed=None):
    if mode == "stream":
        yield from broadcast_tree_stream_data_loader(tokenizer, config, batch_size, seq_len,
                                                     batch_height=batch_height, summary=summary, device=device)
    else:
        yield from broadcast_tree_sample_data_loader(tokenizer, config, batch_size, seq_len,
                                                     batch_height=batch_height, summary=summary, device=device,
                                                     seed=seed)


def block_autoregressive_tree_data_loader(tokenizer: BroadcastTreeTokenizer, config: PerfectTreeConfig,
                                          batch_size, seq_len, batch_height=None, device="cpu"):
    needed_tokens = batch_size * seq_len + 1
    trees = tokenizer.tokenized_markov_subtrees_stream(config, batch_height=batch_height)
    for tokens in uniform_slices_from_concatenation(trees, needed_tokens):
        yield tokens_to_data(tokens, batch_size, seq_len, device, compact=True)
