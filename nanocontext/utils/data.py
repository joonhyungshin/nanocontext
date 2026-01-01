from collections import deque

import torch

from .compute import device_to_use
from .dist import main_process


def uniform_slices_from_concatenation(generator, size):
    token_buffer = deque()
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
