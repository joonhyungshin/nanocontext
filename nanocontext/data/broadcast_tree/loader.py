import torch

from nanocontext.data.common import tokens_to_data
from nanocontext.tree import BroadcastChannel, LazyBroadcastTree, PerfectTreeConfig
from nanocontext.utils import get_numpy_rng, uniform_slices_from_concatenation

from .tokenizer import SummaryTokenizer, PerfectTreeTokenizer


class BroadcastTreeStreamer:
    def __init__(self, tokenizer: PerfectTreeTokenizer, tree_config: PerfectTreeConfig, channel: BroadcastChannel):
        self.tokenizer = tokenizer
        self.channel = channel
        self.tree_config = tree_config

    def tokenized_trees_stream(self, token_start_idx=0, batch_height=None):
        beginning = True
        while True:
            tree = LazyBroadcastTree(self.tree_config, self.channel)
            token_stream = self.tokenizer.tokenize_lazy_stream(tree, token_start_idx=token_start_idx,
                                                               batch_height=batch_height, prepend_bos=True)
            for token, _, __ in token_stream:
                yield token
            if beginning:
                beginning = False
                token_start_idx = 0

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
            tree = LazyBroadcastTree(config, self.channel)
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
                    yield num_trees, tokens_window, summary
                    tokens_window = []
            beginning = False
            num_trees += 1


def broadcast_tree_stream_data_loader(tokenizer: PerfectTreeTokenizer, config: PerfectTreeConfig,
                                      channel: BroadcastChannel, batch_size, seq_len,
                                      batch_height=None, summary_every=None, device="cpu"):
    streamer = BroadcastTreeStreamer(tokenizer, config, channel)
    if summary_every is not None:
        if not isinstance(tokenizer, SummaryTokenizer):
            raise ValueError("tokenizer does not support summarizing")
        stream = streamer.tokenized_trees_with_summaries_stream(summary_every, batch_height=batch_height)
        _, _, last_summary = next(stream)
        while True:
            all_tokens = []
            mask_len = []
            for _ in range(batch_size):
                batch_tokens = last_summary
                mask_len.append([len(batch_tokens) - 1])
                while len(batch_tokens) <= seq_len:
                    _, tokens, summary_write = next(stream)
                    batch_tokens += tokens + summary_write
                    last_summary = summary_write
                all_tokens.extend(batch_tokens[:seq_len + 1])
            x, y = tokens_to_data(all_tokens, batch_size, seq_len, device, compact=False)
            y = y.clone()
            mask = torch.arange(seq_len, device=y.device) < torch.tensor(mask_len, device=y.device)
            y[mask] = -1
            yield x, y
    else:
        needed_tokens = batch_size * seq_len + 1
        trees = streamer.tokenized_trees_stream(batch_height=batch_height)
        for tokens in uniform_slices_from_concatenation(trees, needed_tokens):
            yield tokens_to_data(tokens, batch_size, seq_len, device, compact=True)


def broadcast_tree_sample_data_loader(tokenizer: PerfectTreeTokenizer, config: PerfectTreeConfig,
                                      channel: BroadcastChannel, batch_size, seq_len,
                                      batch_height=None, summary_every=None, device="cpu", seed=None):
    rng = get_numpy_rng(seed, local=True)
    d, height = config.d, config.height
    num_tokens = (d ** (height - 1)) * (d + 1)
    streamer = BroadcastTreeStreamer(tokenizer, config, channel)
    if summary_every is not None:
        if not isinstance(tokenizer, SummaryTokenizer):
            raise ValueError("tokenizer does not support summarizing")
        while True:
            tokens = []
            mask_len = []
            for _ in range(batch_size):
                start_idx = rng.integers(0, num_tokens - 1)
                stream = streamer.tokenized_trees_with_summaries_stream(summary_every,
                                                                        batch_height=batch_height,
                                                                        token_start_idx=start_idx)
                _, _, batch_tokens = next(stream)
                mask_len.append([len(batch_tokens) - 1])
                while len(batch_tokens) <= seq_len:
                    num_trees, cur_tokens, summary_write = next(stream)
                    batch_tokens += cur_tokens + summary_write
                tokens.extend(batch_tokens[:seq_len + 1])
            x, y = tokens_to_data(tokens, batch_size, seq_len, device, compact=False)
            y = y.clone()
            mask = torch.arange(seq_len, device=y.device) < torch.tensor(mask_len, device=y.device)
            y[mask] = -1
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
                               channel: BroadcastChannel, batch_size, seq_len,
                               batch_height=None, mode="stream", summary_every=-1, device="cpu", seed=None):
    if mode == "stream":
        yield from broadcast_tree_stream_data_loader(tokenizer, config, channel, batch_size, seq_len,
                                                     batch_height=batch_height, summary_every=summary_every,
                                                     device=device)
    else:
        yield from broadcast_tree_sample_data_loader(tokenizer, config, channel, batch_size, seq_len,
                                                     batch_height=batch_height, summary_every=summary_every,
                                                     device=device, seed=seed)
