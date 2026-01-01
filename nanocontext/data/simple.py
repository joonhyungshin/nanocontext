import numpy as np

from .common import tokens_to_data


def zero_one_data_loader(batch_size, seq_len, device="cpu", period=2, seed=None):
    needed_tokens = batch_size * seq_len + 1
    tokens = np.zeros(needed_tokens)
    while True:
        rng = np.random.default_rng(seed)
        start = rng.integers(0, period)
        tokens[np.arange(start, needed_tokens, period)] = 1
        yield tokens_to_data(tokens, batch_size, seq_len, device)
