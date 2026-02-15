from collections import deque

import torch

from .compute import device_to_use
from .dist import main_process


def uniform_slices_from_concatenation(generator, size, start_idx=0):
    token_buffer = deque()
    while len(token_buffer) < start_idx:
        token_buffer.extend(next(generator))
    for _ in range(start_idx):
        token_buffer.popleft()
    while True:
        while len(token_buffer) < size:
            token_buffer.extend(next(generator))
        tokens = [token_buffer.popleft() for _ in range(size)]
        yield tokens


@main_process
def save_model(model_data, model_path):
    torch.save(model_data, model_path)


def load_model(model_path):
    model_data = torch.load(model_path, map_location=device_to_use())
    return model_data
