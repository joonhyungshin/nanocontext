from nanocontext.data.common import tokens_to_data
from nanocontext.utils import get_numpy_rng, uniform_slices_from_concatenation

from .tree import BroadcastConfig
from .tokenizer import tokenized_broadcast_trees, tokenized_broadcast_trees_with_summaries


def broadcast_tree_stream_data_loader(config: BroadcastConfig, batch_size, seq_len, batch_height, tokenizer,
                                      summary=False, device="cpu", seed=None):
    rng = get_numpy_rng(seed, local=True)
    if summary:
        needed_tokens = batch_size * (seq_len + 1)
        summary_len = len(tokenizer.init_summary_tokens(config))
        content_len = seq_len + 1 - 2 * summary_len
        if content_len <= 0:
            raise ValueError("context size too small")
        stream = tokenized_broadcast_trees_with_summaries(config, batch_height, tokenizer, content_len, seed=rng)
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
        trees = tokenized_broadcast_trees(config, batch_height, tokenizer, rng)
        for tokens in uniform_slices_from_concatenation(trees, needed_tokens):
            yield tokens_to_data(tokens, batch_size, seq_len, device, compact=True)


def broadcast_tree_sample_data_loader(config, batch_size, seq_len, batch_height, tokenizer,
                                      summary=False, device="cpu", seed=None):
    raise NotImplementedError


def broadcast_tree_data_loader(config: BroadcastConfig, batch_size, seq_len, batch_height, tokenizer,
                               mode="stream", summary=False, device="cpu", seed=None):
    if mode == "stream":
        yield from broadcast_tree_stream_data_loader(config, batch_size, seq_len, batch_height, tokenizer,
                                                     summary=summary, device=device, seed=seed)
    else:
        yield from broadcast_tree_sample_data_loader(config, batch_size, seq_len, batch_height, tokenizer,
                                                     summary=summary, device=device, seed=seed)
