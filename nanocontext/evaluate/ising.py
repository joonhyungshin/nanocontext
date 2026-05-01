from collections import deque
import math

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from nanocontext.data.broadcast_tree import Engine, SummaryTokenizer
from nanocontext.data.broadcast_tree.loader import BroadcastTreeStreamer
from nanocontext.evaluate import infer_summary_every
from nanocontext.tree import PerfectTreeConfig, IsingBroadcastChannel
from nanocontext.utils import ddp_world_size


@torch.inference_mode()
def sample_magnets(engine: Engine, prompt, num_samples, max_tokens,
                   batch_samples=None, max_summary_tokens=64, **kwargs):
    batch_samples = batch_samples or num_samples
    magnet = torch.zeros(num_samples, device=engine.device)
    tokenizer = engine.tokenizer
    pos_token = tokenizer.tokenize_value(1)
    neg_token = tokenizer.tokenize_value(-1)
    for i in range(0, num_samples, batch_samples):
        actual_batch_samples = min(num_samples - i, batch_samples)
        for token_tensor in engine.generate_tree_tokens_tensor_stream(prompt,
                                                                      max_tokens=max_tokens,
                                                                      allow_many=True,
                                                                      num_samples=actual_batch_samples,
                                                                      max_context_tokens=max_summary_tokens,
                                                                      **kwargs):
            spin_tensor = (token_tensor == pos_token).int() - (token_tensor == neg_token).int()
            magnet[i:i + actual_batch_samples] += spin_tensor
    return magnet


def gather_magnets(engine: Engine, prompt, total_samples, max_tokens,
                   batch_samples=None, **kwargs):
    world_size = ddp_world_size()
    num_samples = (total_samples + world_size - 1) // world_size
    total_samples = num_samples * world_size
    magnet = sample_magnets(engine, prompt, num_samples, max_tokens, batch_samples=batch_samples, **kwargs)
    if world_size > 1:
        magnets = torch.empty(total_samples, dtype=magnet.dtype, device=magnet.device)
        dist.all_gather_into_tensor(magnets, magnet)
        return magnets
    else:
        return magnet


def compute_moments(magnet):
    n = len(magnet)
    total_magnet = np.sum(magnet)
    mean_magnet = total_magnet / n
    magnet_var = np.sum((magnet - mean_magnet) ** 2)
    magnet_fourth = np.sum((magnet - mean_magnet) ** 4)
    biased_var = magnet_var / n
    biased_fourth = magnet_fourth / n
    unbiased_var = magnet_var / (n - 1)
    sample_kurtosis = biased_fourth / biased_var ** 2 - 3
    fisher_kurtosis = (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * sample_kurtosis + 6)
    return unbiased_var, fisher_kurtosis


def evaluate_moments(engine: Engine, prompt, total_samples, max_tokens,
                     batch_samples=None, actual_tokens_hint=None, **kwargs):
    """Computes sample variance and excess kurtosis."""
    magnet_tensor = gather_magnets(engine, prompt, total_samples, max_tokens, batch_samples=batch_samples,
                                   **kwargs)
    magnet = magnet_tensor.detach().cpu().numpy()
    normalized_magnet = magnet / math.sqrt(actual_tokens_hint or max_tokens)
    return compute_moments(normalized_magnet)


@torch.inference_mode()
def evaluate_perplexity(engine: Engine, total_samples: int,
                        tree_config: PerfectTreeConfig, channel: IsingBroadcastChannel,
                        batch_samples=None, batch_height=None, summary_every=None):
    tokenizer = engine.tokenizer
    model = engine.model
    context_len = model.config.sequence_len
    world_size = ddp_world_size()
    num_samples = (total_samples + world_size - 1) // world_size
    batch_samples = batch_samples or num_samples
    total_samples = num_samples * world_size
    streamer = BroadcastTreeStreamer(tokenizer, tree_config, channel)
    if not isinstance(tokenizer, SummaryTokenizer):
        summary_every = None
    else:
        prompt = tokenizer.init_summary_tokens(tree_config)
        summary_every = summary_every or infer_summary_every(engine, prompt, tree_config)
        # summary_len = len(tokenizer.init_summary_tokens(tree_config))
        # max_summary_every = context_len + 1 - 2 * summary_len
        # summary_every = min(max_summary_every, summary_every or max_summary_every)

    def data_stream():
        if summary_every is not None:
            for _ in range(num_samples):
                stream = streamer.tokenized_trees_with_summaries_stream(summary_every,
                                                                        batch_height=batch_height, num_trees=1)
                _, __, all_tokens = next(stream)
                summary_len = len(all_tokens)
                for _, tokens, summary in stream:
                    all_tokens += tokens + (summary or [])
                    tokens_len = len(all_tokens)
                    x = all_tokens[:-1] + [0] * (context_len - tokens_len + 1)
                    y = [-1] * (summary_len - 1) + all_tokens[summary_len:] + [-1] * (context_len - tokens_len + 1)
                    yield x, y
                    all_tokens = summary
                    summary_len = len(summary)
        else:
            context_window = deque()
            beginning = True
            for token in streamer.tokenized_trees_stream(batch_height=batch_height, num_trees=num_samples):
                if token == tokenizer.bos_token:
                    if beginning and context_window:
                        buffer = list(context_window)
                        x = buffer[:-1] + [0] * (context_len - len(buffer) + 1)
                        y = buffer[1:] + [-1] * (context_len - len(buffer) + 1)
                        yield x, y
                    beginning = True
                    context_window.clear()
                context_window.append(token)
                if len(context_window) > context_len + 1:
                    context_window.popleft()
                if len(context_window) == context_len + 1:
                    buffer = list(context_window)
                    x = buffer[:-1]
                    if beginning:
                        y = buffer[1:]
                    else:
                        y = [-1] * context_len + [buffer[-1]]
                    yield x, y
                    beginning = False

    def total_cross_entropy(x, y):
        logits = model(x)
        return F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1),
                               ignore_index=-1, reduction="sum")

    x_tensor = torch.empty((batch_samples, context_len), device=engine.device, dtype=torch.long)
    y_tensor = torch.empty((batch_samples, context_len), device=engine.device, dtype=torch.long)
    batch_idx = 0
    perplexity = torch.tensor([0], device=engine.device, dtype=torch.float)
    for x_list, y_list in data_stream():
        x_tensor[batch_idx, :] = torch.tensor(x_list)
        y_tensor[batch_idx, :] = torch.tensor(y_list)
        batch_idx += 1
        if batch_idx >= batch_samples:
            perplexity += total_cross_entropy(x_tensor, y_tensor)
            batch_idx = 0
    if batch_idx > 0:
        perplexity += total_cross_entropy(x_tensor[:batch_idx, :], y_tensor[:batch_idx, :])
    if world_size > 1:
        dist.all_reduce(perplexity, op=dist.ReduceOp.SUM)
    perplexity = perplexity.item() / total_samples
    return perplexity
