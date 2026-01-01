from functools import partial
import time

import click

import torch

from nanocontext.data.broadcast_tree import broadcast_tree_data_loader, decode_trees
from nanocontext.models.nanochat import NanochatConfig, Nanochat
from nanocontext.evaluate import evaluate_var_sum
from nanocontext.train import NanochatTrainerConfig, NanochatTrainer, TrainerSignal
from nanocontext.sample import NanochatSampler
from nanocontext.utils import (ddp_context, ddp_world_size, device_to_use, is_main_process,
                               main_process, save_model, load_model, synchronize, RNGManager)

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
@click.option("--wandb", "wandb_mode", help="wandb mode", default="online",
              type=click.Choice(["online", "offline", "disabled"]))
@click.option("--wandb-log-every", help="wandb log every n steps", default=10, type=int)
@click.option("--buffer-size", help="buffer size to stream a tree", default=1024, type=int)
@click.option("--sample-every", help="sample a tree every few steps", type=int)
@click.option("--eval-every", help="evaluate every few steps", default=20, type=int)
@click.option("--eval-height", help="height to use in evaluation", type=int)
@click.option("--eval-samples", help="number of samples for evaluation", default=32, type=int)
@click.option("--seed", help="random seed", type=int)
def train(d, rho, height, device_batch_size, total_batch_size,
          context_len, vocab_size, layers, heads, kv_heads, model_dim,
          unembedding_lr, embedding_lr, matrix_lr, weight_decay,
          warmup_ratio, warmdown_ratio, final_lr,
          num_iterations, param_data_ratio,
          save_to, wandb_mode, wandb_log_every,
          eval_every, eval_height, eval_samples,
          buffer_size, sample_every, seed):
    rng = RNGManager(seed=seed)
    echo(f"training with seed: {rng.seed}")
    batch_height = 0
    buffer_len = 1
    while buffer_len * d <= buffer_size and batch_height < height:
        batch_height += 1
        buffer_len *= d
    eval_height = eval_height or height
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
                                                device=device, seed=rng.local_numpy_rng)
        wandb_conf = model_kwargs | trainer_kwargs | {
            "d": d,
            "rho": rho,
            "height": height,
            "device_batch_size": device_batch_size,
            "total_batch_size": total_batch_size,
            "eval_height": eval_height,
            "eval_samples": eval_samples,
            "seed": seed,
        }
        ctx = wandb_conf | {
            "total_training_time": 0,
            "smooth_train_loss": 0.0,
        }
        if not is_main_process():
            wandb_mode = "disabled"
        with wandb.init(config=wandb_conf, mode=wandb_mode) as run:
            trainer = NanochatTrainer(trainer_conf, model, dataloader, seed=rng.global_torch_rng)
            trainer.register_callback(TrainerSignal.PRE_OPTIM_STEP,
                                      partial(timer_start, ctx=ctx),
                                      "timer_start")
            trainer.register_callback(TrainerSignal.PRE_OPTIM_STEP,
                                      partial(sample_validate,sample_every, seed=rng.local_torch_rng),
                                      "sample_validate")
            trainer.register_callback(TrainerSignal.PRE_OPTIM_STEP,
                                      partial(evaluate,eval_every, run=run, ctx=ctx, seed=rng.local_torch_rng),
                                      "evaluate")
            trainer.register_callback(TrainerSignal.POST_OPTIM_STEP,
                                      partial(log_trainer_stats, wandb_log_every, run=run, ctx=ctx),
                                      "log_trainer_stats")
            trainer.init_weights()
            trainer.train(num_iterations, grad_accum_steps)
            echo("Training finished")
            echo(f"Elapsed: {ctx["total_training_time"] / 60:.2f}m")
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
    rng = RNGManager(seed=seed)
    echo(f"generating with seed: {rng.seed}")
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
    sampler = NanochatSampler(model, seed=rng.global_torch_rng)
    tokens = sampler.generate_batch([0], **sampler_kwargs)
    for i in range(samples):
        for tree in decode_trees(tokens[i]):
            echo(tree)


def model_hyperparams_from_layers(n_layers, n_heads=None, n_kv_heads=None, n_embd=None):
    if not n_embd:
        n_embd = n_layers * 64
    if not n_heads:
        n_heads = max(1, (n_embd + 127) // 128)
    if not n_kv_heads:
        n_kv_heads = n_heads
    return n_heads, n_kv_heads, n_embd


echo = main_process(click.echo)


def evaluate(evaluate_every, step, num_iterations, model, run, ctx, seed):
    if evaluate_every is not None and (step % evaluate_every == 0 or step == num_iterations):
        d, height, num_samples = ctx["d"], ctx["eval_height"], ctx["num_samples"]
        max_tokens = 1
        for _ in range(height):
            max_tokens = d * max_tokens + d - 1
        model.eval()
        sample_var = evaluate_var_sum(model, num_samples, max_tokens, seed=seed)
        model.train()
        echo(f"Sample variance of magnetization for height {height}: {sample_var:.6f}")
        run.log({
            "sample_variance": sample_var,
        }, step=step)


@main_process
def sample_validate(sample_every, step, num_iterations, model, seed):
    if sample_every is not None and (step % sample_every == 0 or step == num_iterations):
        model.eval()
        sampler = NanochatSampler(model, seed=seed)
        tokens = sampler.generate_batch([0], max_tokens=16, end_token=0)[0]
        if len(tokens) <= 1:
            echo("(empty)")
        else:
            echo(decode_trees(tokens)[0])
        model.train()


def log_trainer_stats(wandb_log_every, run, step, num_iterations, stats, ctx):
    synchronize()
    dt = time.time() - ctx["timer_start"]
    total_training_time = ctx["total_training_time"] + dt
    train_loss = stats["train_loss"]
    ema_beta = 0.9
    smooth_train_loss = ema_beta * ctx["smooth_train_loss"] + (1 - ema_beta) * train_loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    tok_per_sec = int(ctx["total_batch_size"] / dt)
    ctx.update({
        "total_training_time": total_training_time,
        "smooth_train_loss": smooth_train_loss,
    })
    log_data = {
        "train_loss": debiased_smooth_loss,
        "train_lrm": stats["lrm"],
        "train_dt": dt,
        "train_tok_per_sec": tok_per_sec,
        "total_training_time": total_training_time,
    }
    if "grad_norm" in stats:
        log_data["train_grad_norm"] = stats["grad_norm"]
    if step % wandb_log_every == 0:
        run.log(log_data, step=step)
    echo(f"Step {step}/{num_iterations} done "
         f"| loss: {debiased_smooth_loss:.6f} "
         f"| dt: {dt * 1000:.2f}ms "
         f"| elapsed: {total_training_time / 60:.2f}m")


def timer_start(ctx, **_):
    synchronize()
    ctx["timer_start"] = time.time()


cli()
