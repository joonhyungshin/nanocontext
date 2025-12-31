from collections import deque
from contextlib import contextmanager, nullcontext
import functools
import os

import click

import torch
import torch.distributed as dist
import torch.nn.functional as F


def ddp_local_rank():
    return int(os.getenv("LOCAL_RANK", "0"))


def ddp_rank():
    return int(os.getenv("RANK", "0"))


def ddp_world_size():
    return int(os.getenv("WORLD_SIZE", "1"))


def is_main_process():
    return ddp_rank() == 0


def main_process(func=None, return_otherwise=None):
    if func is not None:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if is_main_process():
                return func(*args, **kwargs)
            return return_otherwise
        return wrapper
    else:
        def decorator(f):
            @functools.wraps(f)
            def inner_wrapper(*args, **kwargs):
                if is_main_process():
                    return f(*args, **kwargs)
                return return_otherwise
            return inner_wrapper
        return decorator


def device_to_use():
    if torch.cuda.is_available():
        return torch.device("cuda", ddp_local_rank())
    return torch.device("cpu")


def ddp_setup():
    if dist.is_nccl_available() and torch.cuda.is_available():
        dist.init_process_group(backend="nccl", device_id=device_to_use())
    elif dist.is_gloo_available():
        dist.init_process_group(backend="gloo")


def ddp_teardown():
    dist.destroy_process_group()


@contextmanager
def ddp_context():
    if dist.is_torchelastic_launched():
        ddp_setup()
        dist.barrier()
    try:
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.fp32_precision = "tf32"
            with torch.cuda.device(device_to_use()):
                yield
        else:
            yield
    finally:
        if dist.is_torchelastic_launched():
            ddp_teardown()


def rms_norm(x):
    return F.rms_norm(x, (x.size(-1),))


def rotary_emb_attn(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:d + d]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], dim=3)


def d_order(n, d):
    cnt = 0
    while n % d == 0:
        cnt += 1
        n //= d
    return cnt


@main_process
def echo(*args, **kwargs):
    click.echo(*args, **kwargs)


def autocast():
    device_type = device_to_use().type
    if device_type == "cuda":
        return torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
    return nullcontext()


@main_process
def save_model(model_data, model_path):
    torch.save(model_data, model_path)


def load_model(model_path):
    model_data = torch.load(model_path, map_location=device_to_use())
    return model_data


def uniform_slices_from_concatenation(generator, size):
    token_buffer = deque()
    while True:
        while len(token_buffer) < size:
            token_buffer.extend(next(generator))
        tokens = [token_buffer.popleft() for _ in range(size)]
        yield tokens
