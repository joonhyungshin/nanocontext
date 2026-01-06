import time

import click

import torch

from nanocontext.data.broadcast_tree import broadcast_tree_data_loader, SpinTreeTokenizer
from nanocontext.models.nanochat import NanochatConfig, Nanochat
from nanocontext.evaluate import evaluate_moments, gather_magnets
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
@click.option("--rotary-seq-len", help="rotary sequence length", type=int)
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
@click.option("--sample-max-tokens", help="sample max tokens", default=32, type=int)
@click.option("--eval-every", help="evaluate every few steps", default=20, type=int)
@click.option("--eval-height", help="height to use in evaluation", type=int)
@click.option("--eval-samples", help="number of samples for evaluation", default=32, type=int)
@click.option("--hist-every", help="compute histogram every few steps", default=100, type=int)
@click.option("--hist-height", help="height to use in histogram computation", type=int)
@click.option("--hist-samples", help="number of samples for histogram computation", type=int)
@click.option("--sample-batch", help="batch size for sampling", type=int)
@click.option("--seed", help="random seed", type=int)
def train(d, rho, height, device_batch_size, total_batch_size,
          context_len, vocab_size, layers, heads, kv_heads, model_dim, rotary_seq_len,
          unembedding_lr, embedding_lr, matrix_lr, weight_decay,
          warmup_ratio, warmdown_ratio, final_lr,
          num_iterations, param_data_ratio,
          save_to, wandb_mode, wandb_log_every,
          eval_every, eval_height, eval_samples,
          hist_every, hist_height, hist_samples, sample_batch,
          buffer_size, sample_every, sample_max_tokens, seed):
    rng = RNGManager(seed=seed)
    echo(f"training with seed: {rng.seed}")
    batch_height = 0
    buffer_len = 1
    while buffer_len * d <= buffer_size and batch_height < height:
        batch_height += 1
        buffer_len *= d
    eval_height = eval_height or height
    hist_height = hist_height or eval_height
    hist_samples = hist_samples or eval_samples
    heads, kv_heads, model_dim = model_hyperparams_from_layers(layers, heads, kv_heads, model_dim)
    rotary_seq_len = rotary_seq_len or context_len * 10
    model_kwargs = dict(sequence_len=context_len, vocab_size=vocab_size, rotary_seq_len=rotary_seq_len,
                        n_layers=layers, n_heads=heads, n_kv_heads=kv_heads, n_embd=model_dim)
    trainer_kwargs = dict(unembedding_lr=unembedding_lr, embedding_lr=embedding_lr, matrix_lr=matrix_lr,
                          weight_decay=weight_decay,
                          warmup_ratio=warmup_ratio, warmdown_ratio=warmdown_ratio,
                          final_lr_frac=final_lr)
    model_conf = NanochatConfig(**model_kwargs)
    trainer_conf = NanochatTrainerConfig(**trainer_kwargs)
    tokenizer = SpinTreeTokenizer(vocab_size)
    with ddp_context():
        device = device_to_use()
        world_size = ddp_world_size()
        world_tokens = device_batch_size * context_len * world_size
        eval_samples = (eval_samples + world_size - 1) // world_size * world_size
        hist_samples = (hist_samples + world_size - 1) // world_size * world_size
        grad_accum_steps = total_batch_size // world_tokens
        with torch.device("meta"):
            model = Nanochat(model_conf)
        model.to_empty(device=device)
        if not num_iterations:
            num_params = sum(p.numel() for p in model.parameters())
            target_tokens = param_data_ratio * num_params
            num_iterations = target_tokens // total_batch_size
        dataloader = broadcast_tree_data_loader(d, rho, height,
                                                device_batch_size, context_len, batch_height, tokenizer,
                                                device=device, seed=rng.local_numpy_rng)
        wandb_conf = model_kwargs | trainer_kwargs | {
            "d": d,
            "rho": rho,
            "height": height,
            "device_batch_size": device_batch_size,
            "total_batch_size": total_batch_size,
            "eval_height": eval_height,
            "eval_samples": eval_samples,
            "hist_height": hist_height,
            "hist_samples": hist_samples,
            "seed": rng.seed,
        }
        ctx = wandb_conf | {
            "total_training_time": 0,
            "smooth_train_loss": 0.0,
            "sample_batch": sample_batch,
        }
        if not is_main_process():
            wandb_mode = "disabled"
        with wandb.init(config=wandb_conf, mode=wandb_mode) as run:
            trainer = NanochatTrainer(trainer_conf, model, dataloader, seed=rng.global_torch_rng)
            trainer.register_callback(TrainerSignal.PRE_OPTIM_STEP,
                                      sample_validate,
                                      sample_every, sample_max_tokens, tokenizer,
                                      seed=rng.local_torch_rng)
            trainer.register_callback(TrainerSignal.PRE_OPTIM_STEP,
                                      evaluate,
                                      eval_every, tokenizer,
                                      run=run, ctx=ctx, seed=rng.local_torch_rng)
            trainer.register_callback(TrainerSignal.PRE_OPTIM_STEP,
                                      histogram,
                                      hist_every, tokenizer,
                                      run=run, ctx=ctx, seed=rng.local_torch_rng)
            trainer.register_callback(TrainerSignal.PRE_OPTIM_STEP, timer_start, ctx=ctx)
            trainer.register_callback(TrainerSignal.POST_OPTIM_STEP,
                                      log_trainer_stats, wandb_log_every, run=run, ctx=ctx)
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
@click.option("--rotary-seq-len", help="rotary sequence length", type=int)
@click.option("--max-tokens", help="maximum number of tokens", type=int, required=True)
@click.option("--temperature", help="sampling temperature", default=1.0, type=float)
@click.option("--top-k", help="top-k sampling", type=int)
@click.option("--samples", help="number of samples to generate", default=1, type=int)
@click.option("--model-path", help="path to model", type=str, required=True)
@click.option("--seed", help="random seed", type=int)
def generate(context_len, vocab_size, layers, heads, kv_heads, model_dim, rotary_seq_len,
             max_tokens, temperature, top_k, samples,
             model_path, seed):
    rng = RNGManager(seed=seed)
    echo(f"generating with seed: {rng.seed}")
    heads, kv_heads, model_dim = model_hyperparams_from_layers(layers, heads, kv_heads, model_dim)
    rotary_seq_len = rotary_seq_len or context_len * 10
    model_kwargs = dict(sequence_len=context_len, vocab_size=vocab_size, rotary_seq_len=rotary_seq_len,
                        n_layers=layers, n_heads=heads, n_kv_heads=kv_heads, n_embd=model_dim)
    sampler_kwargs = dict(num_samples=samples,
                          max_tokens=max_tokens, end_token=0, temperature=temperature, top_k=top_k)
    tokenizer = SpinTreeTokenizer(vocab_size)
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
        for tree in tokenizer.decode_trees(tokens[i]):
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


def get_max_tokens(d, height):
    max_tokens = d
    for _ in range(height - 1):
        max_tokens = d * max_tokens + d - 1
    return max_tokens


def evaluate(evaluate_every, tokenizer, step, num_iterations, model, run, ctx, seed):
    if evaluate_every is not None and (step % evaluate_every == 0 or step == num_iterations):
        d, height, total_samples, batch_samples = (ctx["d"], ctx["eval_height"],
                                                   ctx["eval_samples"], ctx["sample_batch"])
        max_tokens = get_max_tokens(d, height)
        actual_tokens = d ** height
        model.eval()
        sample_var, kurtosis = evaluate_moments(model, tokenizer, total_samples, max_tokens,
                                                batch_samples=batch_samples, actual_tokens_hint=actual_tokens,
                                                seed=seed)
        model.train()
        echo(f"Samples of scaled magnetization for height {height}: var {sample_var:.6f} and kurtosis {kurtosis:.6f}")
        run.log({
            "sample_variance": sample_var * actual_tokens,
            "scaled_sample_variance": sample_var,
            "sample_kurtosis": kurtosis,
        }, step=step)


def histogram(hist_every, tokenizer, step, num_iterations, model, run, ctx, seed):
    if hist_every is not None and (step % hist_every == 0 or step == num_iterations):
        d, height, total_samples, batch_samples = (ctx["d"], ctx["hist_height"],
                                                   ctx["hist_samples"], ctx["sample_batch"])
        max_tokens = get_max_tokens(d, height)
        model.eval()
        magnets = gather_magnets(
            model, tokenizer, total_samples, max_tokens,
            batch_samples=batch_samples, seed=seed
        ).detach().cpu().numpy()
        model.train()
        run.log({
            "magnets": wandb.Histogram(magnets),
        }, step=step)


@main_process
def sample_validate(sample_every, sample_max_tokens, tokenizer, step, num_iterations, model, seed):
    if sample_every is not None and (step % sample_every == 0 or step == num_iterations):
        model.eval()
        sampler = NanochatSampler(model, seed=seed)
        tokens = sampler.generate_batch([0], max_tokens=sample_max_tokens, end_token=0)[0]
        if len(tokens) <= 1:
            echo("(empty)")
        else:
            echo(next(tokenizer.decode_trees(tokens)))
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
