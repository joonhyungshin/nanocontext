from contextlib import contextmanager
import functools
import os

import torch
import torch.distributed as dist


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


def device_to_use():
    if torch.cuda.is_available():
        return torch.device("cuda", ddp_local_rank())
    return torch.device("cpu")
