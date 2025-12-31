import click

import numpy as np
import torch
import secrets

from nanocontext.data.broadcast_tree import broadcast_tree_data_loader, decode_trees
from nanocontext.models.nanochat import NanochatConfig, Nanochat
from nanocontext.train import NanochatTrainerConfig, NanochatTrainer
from nanocontext.sample import NanochatSampler
from nanocontext.utils import (ddp_context, ddp_rank, ddp_world_size, device_to_use,
                               echo, save_model, load_model, get_seeds)

import wandb


@click.group()
def cli():
    pass


@cli.command()
@click.option("-d", help="number of children of a tree", default=3, type=int)
@click.option("--rho", help="correlation", type=float, required=True)
@click.option("--height", help="height of a tree", type=int, required=True)
@click.option("--device-batch-size", help="batch size per device", default=32, type=int)
@click.option("--total-batch-size", help="total batch size in training", default=524288, type=int)
@click.option("--context-size", "context_len", help="context length", default=2048, type=int)
@click.option("--vocab-size", help="vocab size", default=32, type=int)
@click.option("--layers", help="number of layers", default=20, type=int)
@click.option("--heads", help="number of heads", type=int)
@click.option("--kv-heads", help="number of key-value heads", type=int)
@click.option("--model-dim", help="model dimension", type=int)
@click.option("--unembedding-lr", help="unembedding learning rate", default=0.004, type=float)
@click.option("--embedding-lr", help="embedding learning rate", default=0.2, type=float)
@click.option("--matrix-lr", help="matrix learning rate", default=0.02, type=float)
@click.option("--weight-decay", help="weight decay", default=0.0, type=float)
@click.option("--warmup-ratio", help="warmup ratio in LR scheduler", default=0.0, type=float)
@click.option("--warmdown-ratio", help="warmdown ratio in LR scheduler", default=0.2, type=float)
@click.option("--final-lr", help="final learning rate in LR scheduler", default=0.0, type=float)
@click.option("--num-iterations", help="number of iterations", type=int)
@click.option("--param-data-ratio", help="parameter:data ratio", default=20, type=int)
@click.option("--save-to", help="path to save model", type=str, required=True)
@click.option("--buffer-size", help="buffer size to stream a tree", default=1024, type=int)
@click.option("--seed", help="random seed", type=int)
def train(d, rho, height, device_batch_size, total_batch_size,
          context_len, vocab_size, layers, heads, kv_heads, model_dim,
          unembedding_lr, embedding_lr, matrix_lr, weight_decay,
          warmup_ratio, warmdown_ratio, final_lr,
          num_iterations, param_data_ratio,
          save_to,
          buffer_size, seed):
    seed, torch_seed = get_seeds(seed)
    echo(f"training with seed: {seed}")
    batch_height = 0
    buffer_len = 1
    while buffer_len * d <= buffer_size and batch_height < height:
        batch_height += 1
        buffer_len *= d
    heads, kv_heads, model_dim = model_hyperparams_from_layers(layers, heads, kv_heads, model_dim)
    model_kwargs = dict(sequence_len=context_len, vocab_size=vocab_size,
                        n_layers=layers, n_heads=heads, n_kv_heads=kv_heads, n_embd=model_dim)
    trainer_kwargs = dict(unembedding_lr=unembedding_lr, embedding_lr=embedding_lr, matrix_lr=matrix_lr,
                          weight_decay=weight_decay,
                          warmup_ratio=warmup_ratio, warmdown_ratio=warmdown_ratio,
                          final_lr_frac=final_lr)
    model_conf = NanochatConfig(**model_kwargs)
    trainer_conf = NanochatTrainerConfig(**trainer_kwargs)
    with ddp_context():
        rng = np.random.default_rng(seed=[ddp_rank(), seed])
        device = device_to_use()
        world_tokens = device_batch_size * context_len * ddp_world_size()
        grad_accum_steps = total_batch_size // world_tokens
        with torch.device("meta"):
            model = Nanochat(model_conf)
        model.to_empty(device=device)
        if not num_iterations:
            num_params = sum(p.numel() for p in model.parameters())
            target_tokens = param_data_ratio * num_params
            num_iterations = target_tokens // total_batch_size
        dataloader = broadcast_tree_data_loader(d, rho, height,
                                                device_batch_size, context_len, batch_height, vocab_size,
                                                device=device, seed=rng)
        trainer = NanochatTrainer(trainer_conf, model, dataloader, seed=torch_seed)
        trainer.init_weights()
        trainer.train(num_iterations, grad_accum_steps)
        save_model(model.state_dict(), save_to)


@cli.command()
@click.option("--context-size", "context_len", help="context length", default=2048, type=int)
@click.option("--vocab-size", help="vocab size", default=32, type=int)
@click.option("--layers", help="number of layers", default=20, type=int)
@click.option("--heads", help="number of heads", type=int)
@click.option("--kv-heads", help="number of key-value heads", type=int)
@click.option("--model-dim", help="model dimension", type=int)
@click.option("--max-tokens", help="maximum number of tokens", type=int, required=True)
@click.option("--temperature", help="sampling temperature", default=1.0, type=float)
@click.option("--top-k", help="top-k sampling", type=int)
@click.option("--samples", help="number of samples to generate", default=1, type=int)
@click.option("--model-path", help="path to model", type=str, required=True)
@click.option("--seed", help="random seed", type=int)
def generate(context_len, vocab_size, layers, heads, kv_heads, model_dim,
             max_tokens, temperature, top_k, samples,
             model_path, seed):
    seed, torch_seed = get_seeds(seed)
    echo(f"generating with seed: {seed}")
    heads, kv_heads, model_dim = model_hyperparams_from_layers(layers, heads, kv_heads, model_dim)
    model_kwargs = dict(sequence_len=context_len, vocab_size=vocab_size,
                        n_layers=layers, n_heads=heads, n_kv_heads=kv_heads, n_embd=model_dim)
    sampler_kwargs = dict(num_samples=samples,
                          max_tokens=max_tokens, end_token=0, temperature=temperature, top_k=top_k)
    model_conf = NanochatConfig(**model_kwargs)
    with torch.device("meta"):
        model = Nanochat(model_conf)
    device = device_to_use()
    model.to_empty(device=device)
    model_data = load_model(model_path)
    model.load_state_dict(model_data, strict=True, assign=True)
    model.preprocess()
    model.eval()
    sampler = NanochatSampler(model, seed=torch_seed)
    generated_tokens = [[0] for _ in range(samples)]
    for tokens in sampler.generate([0], **sampler_kwargs):
        for i in range(samples):
            if tokens[i] != 0:
                generated_tokens[i].append(tokens[i])
    for i in range(samples):
        for tree in decode_trees(generated_tokens[i]):
            tree.print_tree()


def model_hyperparams_from_layers(n_layers, n_heads=None, n_kv_heads=None, n_embd=None):
    if not n_embd:
        n_embd = n_layers * 64
    if not n_heads:
        n_heads = max(1, (n_embd + 127) // 128)
    if not n_kv_heads:
        n_kv_heads = n_heads
    return n_heads, n_kv_heads, n_embd


cli()
