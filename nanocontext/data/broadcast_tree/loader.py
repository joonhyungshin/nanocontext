from nanocontext.data.common import tokens_to_data
from nanocontext.utils import get_numpy_rng, uniform_slices_from_concatenation

from .tokenizer import tokenized_broadcast_trees, tokenized_broadcast_trees_with_summaries


def broadcast_tree_stream_data_loader(d, rho, height, batch_size, seq_len, batch_height, tokenizer,
                                      summary=False, device="cpu", seed=None):
    rng = get_numpy_rng(seed, local=True)
    if summary:
        needed_tokens = batch_size * (seq_len + 1)
        summary_len = (d + 1) * height + 2
        content_len = seq_len + 1 - 2 * summary_len
        if content_len <= 0:
            raise ValueError("context size too small")
        while True:
            stream = tokenized_broadcast_trees_with_summaries(d, rho, height, batch_size, tokenizer, content_len,
                                                              seed=rng)
            _, _, all_tokens = next(stream)
            for num_trees, tokens, summary_write in stream:
                all_tokens += tokens + summary_write
                if len(all_tokens) == needed_tokens:
                    x, y = tokens_to_data(all_tokens, batch_size, seq_len, device, compact=False)
                    y = y.clone()
                    y[:, :summary_len - 1] = -1
                    yield x, y
                    # Reset every tree or completed training data
                    if num_trees >= 1:
                        break
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
