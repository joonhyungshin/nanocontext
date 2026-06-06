import time

import click
import torch

from nanocontext.data.broadcast_tree import (
    broadcast_tree_data_loader, SimpleEngine, StatefulEngine, save_engine, load_engine,
    SummaryTokenizer, SegmentSummaryTokenizer, PathSummaryTokenizer,
    PerfectTreeTokenizer
)
from nanocontext.models.nanochat import NanochatConfig, Nanochat
from nanocontext.evaluate.coloring import check_validity
from nanocontext.evaluate.ising import evaluate_moments, gather_magnets
from nanocontext.train import NanochatTrainerConfig, NanochatTrainer, TrainerSignal
from nanocontext.sample import NanochatSampler
from nanocontext.tree import IsingBroadcastChannel, ColoringBroadcastChannel, PerfectTreeConfig
from nanocontext.utils import (
    ddp_context, ddp_world_size, device_to_use, is_main_process,
    main_process, synchronize, RNGManager
)

import wandb

from .common import echo, make_prompt, get_max_tokens, display_recon_stat


@click.command()
@click.option("-d", help="number of children of a tree", type=int, required=True)
@click.option("--rho", help="correlation for Ising experiment", type=float)
@click.option("-k", "--colors", "k", help="number of colors for coloring experiment", type=int)
@click.option("--height", help="height of a tree", type=int, required=True)
@click.option("--device-batch-size", help="batch size per device", default=32, type=int)
@click.option("--total-batch-size", help="total batch size in training", default=524288, type=int)
@click.option("--context-size", "context_len", help="context length", default=2048, type=int)
@click.option("--vocab-size", help="vocab size", default=64, type=int)
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
@click.option("--wandb", "wandb_mode", help="wandb mode", default="disabled",
              type=click.Choice(["online", "offline", "disabled"]))
@click.option("--wandb-log-every", help="wandb log every n steps", default=10, type=int)
@click.option("--batch-height", help="batch height to stream a tree", type=int)
@click.option("--sample-every", help="sample a tree every few steps", type=int)
@click.option("--sample-max-tokens", help="sample max tokens", default=32, type=int)
@click.option("--eval-every", help="evaluate every few steps", default=20, type=int)
@click.option("--eval-height", help="height to use in evaluation", type=int)
@click.option("--eval-samples", help="number of samples for evaluation", default=32, type=int)
@click.option("--hist-every", help="compute histogram every few steps", default=100, type=int)
@click.option("--hist-height", help="height to use in histogram computation", type=int)
@click.option("--hist-samples", help="number of samples for histogram computation", type=int)
@click.option("--sample-batch", help="batch size for sampling", type=int)
@click.option("--data", "data_mode", help="data mode", default="stream",
              type=click.Choice(["stream", "sample"]))
@click.option("--summary", "summary_mode", help="summary mode for training", default="disabled",
              type=click.Choice(["disabled", "segment", "path"]))
@click.option("--summary-every", help="summarize every few steps", type=int)
@click.option("--seed", help="random seed", type=int)
def train(d, rho, k, height, device_batch_size, total_batch_size,
          context_len, vocab_size, layers, heads, kv_heads, model_dim, rotary_seq_len,
          unembedding_lr, embedding_lr, matrix_lr, weight_decay,
          warmup_ratio, warmdown_ratio, final_lr,
          num_iterations, param_data_ratio,
          save_to, wandb_mode, wandb_log_every,
          eval_every, eval_height, eval_samples,
          hist_every, hist_height, hist_samples, sample_batch,
          batch_height, sample_every, sample_max_tokens,
          data_mode, summary_mode, summary_every, seed):
    if (rho is None and k is None) or not (rho is None or k is None):
        raise ValueError("exactly one of k (coloring) or rho (Ising) must be given")
    rng = RNGManager(seed=seed)
    echo(f"training with seed: {rng.seed}")
    batch_height = batch_height or height
    eval_height = eval_height or height
    hist_height = hist_height or eval_height
    eval_height = min(eval_height, height)
    hist_height = min(hist_height, height)
    hist_samples = hist_samples or eval_samples
    heads, kv_heads, model_dim = model_hyperparams_from_layers(layers, heads, kv_heads, model_dim)
    rotary_seq_len = rotary_seq_len or context_len * 10
    tree_kwargs = dict(d=d, height=height)
    model_kwargs = dict(sequence_len=context_len, vocab_size=vocab_size, rotary_seq_len=rotary_seq_len,
                        n_layers=layers, n_heads=heads, n_kv_heads=kv_heads, n_embd=model_dim)
    trainer_kwargs = dict(unembedding_lr=unembedding_lr, embedding_lr=embedding_lr, matrix_lr=matrix_lr,
                          weight_decay=weight_decay,
                          warmup_ratio=warmup_ratio, warmdown_ratio=warmdown_ratio,
                          final_lr_frac=final_lr)
    model_conf = NanochatConfig(**model_kwargs)
    trainer_conf = NanochatTrainerConfig(**trainer_kwargs)
    tree_conf = PerfectTreeConfig(**tree_kwargs)
    with ddp_context():
        channel, channel_conf = get_channel(rho, k, rng.local_numpy_rng)
        tokenizer = get_tokenizer(summary_mode, vocab_size, channel.get_state_space())
        if summary_mode == "disabled" or not isinstance(tokenizer, SummaryTokenizer):
            summary_len = None
            summary_every = None
        else:
            summary_len = len(tokenizer.init_summary_tokens(tree_conf))
            max_summary_every = context_len + 1 - 2 * summary_len
            summary_every = min(max_summary_every, summary_every or max_summary_every)
            if summary_every < 1:
                raise ValueError("context size too small for summary")
        prompt = make_prompt(tokenizer, tree_conf)
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
        dataloader = get_dataloader(tree_conf, channel,
                                    device_batch_size, context_len, batch_height, tokenizer,
                                    summary_every=summary_every, data_mode=data_mode, device=device,
                                    seed=rng.local_numpy_rng)
        sampler = NanochatSampler(model, max_context_len=context_len, seed=rng.local_torch_rng(device))
        engine = get_engine(tokenizer, sampler, summary_len, summary_every)
        wandb_conf = model_kwargs | trainer_kwargs | channel_conf | {
            "d": d,
            "height": height,
            "device_batch_size": device_batch_size,
            "total_batch_size": total_batch_size,
            "eval_height": eval_height,
            "eval_samples": eval_samples,
            "hist_height": hist_height,
            "hist_samples": hist_samples,
            "summary_mode": summary_mode,
            "summary_every": summary_every,
            "batch_height": batch_height,
            "data_mode": data_mode,
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
            trainer = NanochatTrainer(trainer_conf, model, dataloader, seed=rng.global_torch_rng(device))
            trainer.register_callback(TrainerSignal.PRE_OPTIM_STEP,
                                      sample_validate,
                                      sample_every, sample_max_tokens, engine, prompt,
                                      num_iterations=num_iterations, ctx=ctx)
            trainer.register_callback(TrainerSignal.PRE_OPTIM_STEP,
                                      evaluate_model,
                                      eval_every, engine, prompt,
                                      num_iterations=num_iterations, run=run, ctx=ctx)
            trainer.register_callback(TrainerSignal.PRE_OPTIM_STEP,
                                      histogram,
                                      hist_every, engine, prompt,
                                      num_iterations=num_iterations, run=run, ctx=ctx)
            trainer.register_callback(TrainerSignal.PRE_OPTIM_STEP, timer_start, ctx=ctx)
            trainer.register_callback(TrainerSignal.POST_OPTIM_STEP,
                                      log_trainer_stats, wandb_log_every, run=run, ctx=ctx)
            trainer.init_weights()
            trainer.train(num_iterations, grad_accum_steps)
            echo("Training finished")
            echo(f"Elapsed: {ctx['total_training_time'] / 60:.2f}m")
            save_engine(engine, save_to)
            echo(f"Model saved to: {save_to}")


def model_hyperparams_from_layers(n_layers, n_heads=None, n_kv_heads=None, n_embd=None):
    if not n_embd:
        n_embd = n_layers * 64
    if not n_heads:
        n_heads = max(1, (n_embd + 127) // 128)
    if not n_kv_heads:
        n_kv_heads = n_heads
    return n_heads, n_kv_heads, n_embd


def get_channel(rho, k, rng):
    if rho is not None:
        return IsingBroadcastChannel(rho, seed=rng), {
            "policy": "Ising",
            "rho": rho,
        }
    return ColoringBroadcastChannel(k, seed=rng), {
        "policy": "coloring",
        "k": k,
    }


def get_tokenizer(summary_mode, vocab_size, value_space):
    if summary_mode == "segment":
        return SegmentSummaryTokenizer(vocab_size, value_space)
    elif summary_mode == "path":
        return PathSummaryTokenizer(vocab_size, value_space)
    return PerfectTreeTokenizer(vocab_size, value_space)


def get_dataloader(config: PerfectTreeConfig, channel, device_batch_size, context_len, batch_height, tokenizer,
                   data_mode="stream", summary_every=None, device="cpu", seed=None):
    dataloader_kwargs = dict(tokenizer=tokenizer, config=config, channel=channel, batch_size=device_batch_size,
                             seq_len=context_len, batch_height=batch_height, mode=data_mode, device=device,
                             seed=seed)
    return broadcast_tree_data_loader(**dataloader_kwargs, summary_every=summary_every)


def get_engine(tokenizer: PerfectTreeTokenizer, sampler, summary_len, content_len):
    if isinstance(tokenizer, SummaryTokenizer):
        return StatefulEngine(tokenizer, sampler, summary_len=summary_len, content_len=content_len)
    return SimpleEngine(tokenizer, sampler)


@main_process
def sample_validate(sample_every, sample_max_tokens, engine, prompt, step, num_iterations, model, ctx):
    if sample_every is not None and (step % sample_every == 0 or step == num_iterations):
        model.eval()
        echo("Plain sample:")
        tree = engine.generate_tree(prompt, sample_max_tokens, max_context_tokens=64)[0]
        if tree.is_singleton():
            echo("(empty)")
        else:
            echo(tree)
        d, height = ctx["d"], ctx["eval_height"]
        echo(f"Patched sample (d={d}, h={height}):")
        tree_config = PerfectTreeConfig(d=d, height=height)
        tree = engine.generate_patched_tree(prompt, sample_max_tokens, tree_config)[0]
        if tree.is_singleton():
            echo("(empty)")
        else:
            echo(tree)
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


def evaluate_model(evaluate_every, engine, prompt, step, num_iterations, model, run, ctx):
    if evaluate_every is not None and (step % evaluate_every == 0 or step == num_iterations):
        d, height, total_samples, batch_samples = (ctx["d"], ctx["eval_height"],
                                                   ctx["eval_samples"], ctx["sample_batch"])
        if ctx["policy"] == "Ising":
            evaluate_ising(engine, prompt, step, model, run, d, height, total_samples, batch_samples)
        else:
            evaluate_coloring(engine, prompt, step, model, run, d, height, total_samples, batch_samples)


def histogram(hist_every, engine, prompt, step, num_iterations, model, run, ctx):
    if ctx["policy"] == "coloring":
        # No histogram for coloring
        return
    if hist_every is not None and (step % hist_every == 0 or step == num_iterations):
        d, height, total_samples, batch_samples = (ctx["d"], ctx["hist_height"],
                                                   ctx["hist_samples"], ctx["sample_batch"])
        max_tokens = get_max_tokens(d, height)
        model.eval()
        magnets = gather_magnets(
            engine, prompt, total_samples, max_tokens,
            batch_samples=batch_samples
        ).detach().cpu().numpy()
        model.train()
        run.log({
            "magnets": wandb.Histogram(magnets),
        }, step=step)


def evaluate_ising(engine, prompt, step, model, run, d, height, total_samples, batch_samples):
    max_tokens = get_max_tokens(d, height)
    actual_tokens = d ** height
    model.eval()
    sample_var, kurtosis = evaluate_moments(engine, prompt, total_samples, max_tokens,
                                            batch_samples=batch_samples, actual_tokens_hint=actual_tokens)
    model.train()
    echo(f"Samples of scaled magnetization for height {height}: var {sample_var:.6f} and kurtosis {kurtosis:.6f}")
    run.log({
        "sample_variance": sample_var * actual_tokens,
        "scaled_sample_variance": sample_var,
        "sample_kurtosis": kurtosis,
    }, step=step)


def evaluate_coloring(engine, prompt, step, model, run, d, height, total_samples, batch_samples):
    max_tokens = get_max_tokens(d, height)
    config = PerfectTreeConfig(d=d, height=height)
    model.eval()

    checker_kwargs = dict(engine=engine, prompt=prompt, total_samples=total_samples, max_tokens=max_tokens,
                          config=config, batch_samples=batch_samples)
    plain_stat = check_validity(**checker_kwargs)
    patched_stat = check_validity(**checker_kwargs, patch=True)
    model.train()
    echo("Reconstruction statistics (plain)")
    display_recon_stat(plain_stat)
    echo(f"Reconstruction statistics (patched: d={d}, h={height})")
    display_recon_stat(patched_stat)

    valid_rate = (plain_stat["constrained"] + plain_stat["free"]) / total_samples
    patched_valid_rate = (patched_stat["constrained"] + patched_stat["free"]) / total_samples
    run.log({
        "valid_rate": valid_rate,
        "patched_valid_rate": patched_valid_rate,
    }, step=step)
