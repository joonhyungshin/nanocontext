import math
import time

import click
import numpy as np
import torch

from nanocontext.data.broadcast_tree import (
    broadcast_tree_data_loader, SimpleEngine, StatefulEngine, save_engine, load_engine,
    SummaryTokenizer, SegmentSummaryTokenizer, PathSummaryTokenizer,
    PerfectTreeTokenizer
)
from nanocontext.models.nanochat import NanochatConfig, Nanochat
from nanocontext.evaluate.coloring import check_validity, get_root_constraint, UnsatisfiedException
from nanocontext.evaluate.ising import evaluate_moments, gather_magnets, compute_moments
from nanocontext.train import NanochatTrainerConfig, NanochatTrainer, TrainerSignal
from nanocontext.sample import NanochatSampler
from nanocontext.tree import IsingBroadcastPolicy, ColoringBroadcastPolicy, PerfectTreeConfig, InferenceTree
from nanocontext.tree.broadcast import markov_forest
from nanocontext.tree.coloring import ColoringDomain
from nanocontext.tree.ising import IsingDomain
from nanocontext.utils import (
    ddp_context, ddp_world_size, device_to_use, is_main_process, compute_moments,
    main_process, synchronize, RNGManager
)

import wandb


@click.group()
def cli():
    pass


@cli.command()
@click.option("-d", help="number of children of a tree", default=3, type=int)
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
@click.option("--wandb", "wandb_mode", help="wandb mode", default="online",
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
          data_mode, summary_mode, seed):
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
    enable_summary = summary_mode != "disabled"
    with ddp_context():
        policy, policy_conf = get_policy(rho, k, rng.local_numpy_rng)
        tokenizer = get_tokenizer(summary_mode, vocab_size, policy.get_domain())
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
        dataloader = get_dataloader(tree_conf, policy,
                                    device_batch_size, context_len, batch_height, tokenizer,
                                    summary=enable_summary, data_mode=data_mode, device=device,
                                    seed=rng.local_numpy_rng)
        sampler = NanochatSampler(model, max_context_len=context_len, seed=rng.local_torch_rng(device))
        engine = get_engine(tokenizer, sampler)
        wandb_conf = model_kwargs | trainer_kwargs | policy_conf | {
            "d": d,
            "height": height,
            "device_batch_size": device_batch_size,
            "total_batch_size": total_batch_size,
            "eval_height": eval_height,
            "eval_samples": eval_samples,
            "hist_height": hist_height,
            "hist_samples": hist_samples,
            "summary_mode": summary_mode,
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


@cli.command()
@click.option("-d", help="number of children of a tree", type=int, required=True)
@click.option("--height", help="desired height of generated tree", type=int, required=True)
@click.option("--max-tokens", help="maximum number of tokens", type=int, required=True)
@click.option("--temperature", help="sampling temperature", default=1.0, type=float)
@click.option("--top-k", help="top-k sampling", type=int)
@click.option("--samples", help="number of samples to generate", default=1, type=int)
@click.option("--model-path", help="path to model", type=str, required=True)
@click.option("--patch", help="patch the generated tree", is_flag=True)
@click.option("--seed", help="random seed", type=int)
def generate(d, height,
             max_tokens, temperature, top_k, samples,
             model_path, patch, seed):
    rng = RNGManager(seed=seed)
    echo(f"generating with seed: {rng.seed}")
    echo(f"using model: {model_path}")
    tree_kwargs = dict(d=d, height=height)
    gen_kwargs = dict(num_samples=samples, temperature=temperature, top_k=top_k)
    tree_conf = PerfectTreeConfig(**tree_kwargs)
    device = device_to_use()
    engine = load_engine(model_path, device, seed=rng.global_torch_rng(device))
    engine.model.eval()
    prompt = make_prompt(engine.tokenizer, tree_conf)
    if patch:
        tree_generator = engine.generate_patched_tree(prompt, max_tokens, tree_conf, allow_many=True, **gen_kwargs)
    else:
        tree_generator = engine.generate_tree(prompt, max_tokens, **gen_kwargs)
    for tree in tree_generator:
        echo(tree)


@cli.command()
@click.option("-d", help="number of children of a tree", default=3, type=int)
@click.option("--height", help="height of a tree", type=int, required=True)
@click.option("--eval-height", help="height to use in evaluation", type=int)
@click.option("--samples", help="number of samples to generate", default=1024, type=int)
@click.option("--sample-batch", help="batch size for sampling", type=int)
@click.option("--model-path", help="path to model", type=str, required=True)
@click.option("--seed", help="random seed", type=int)
def evaluate(d, height, eval_height, samples, sample_batch, model_path, seed):
    rng = RNGManager(seed=seed)
    echo(f"evaluating with seed: {rng.seed}")
    echo(f"using model: {model_path}")
    eval_height = min(eval_height, height) if eval_height is not None else height
    tree_conf = PerfectTreeConfig(d=d, height=height)
    eval_tree_conf = PerfectTreeConfig(d=d, height=eval_height)

    with ddp_context():
        device = device_to_use()
        engine = load_engine(model_path, device, seed=rng.global_torch_rng(device))
        engine.model.eval()
        domain = engine.tokenizer.domain
        prompt = make_prompt(engine.tokenizer, tree_conf)
        max_tokens = get_max_tokens(d, eval_height)

        echo(f"generating {samples} samples...")
        if isinstance(domain, IsingDomain):
            echo("detected Ising experiment.")
            var, kurtosis = evaluate_moments(engine, prompt, samples, max_tokens,
                                             batch_samples=sample_batch, actual_tokens_hint=d**eval_height)
            echo(f"Variance: {var}")
            echo(f"Kurtosis: {kurtosis}")
        elif isinstance(domain, ColoringDomain):
            echo("detected Coloring experiment.")
            stat = check_validity(engine, prompt, samples, max_tokens, eval_tree_conf,
                                  batch_samples=sample_batch, patch=True)
            display_recon_stat(stat)
            echo(f"Valid rate: {stat['constrained'] + stat['free']} / {samples}")
        else:
            raise click.ClickException("Unknown domain type")


@cli.command()
@click.option("-d", help="number of children of a tree", default=3, type=int)
@click.option("--rho", help="correlation for Ising experiment", type=float)
@click.option("-k", "--colors", "k", help="number of colors for coloring experiment", type=int)
@click.option("--height", help="height of a tree", type=int, required=True)
@click.option("--samples", help="number of samples to generate", default=1024, type=int)
@click.option("--markov-height", help="height when sampled using the Markov process", type=int)
@click.option("--seed", help="random seed", type=int)
def simulate(d, height, rho, k, samples, markov_height, seed):
    if (rho is None and k is None) or not (rho is None or k is None):
        raise ValueError("exactly one of k (coloring) or rho (Ising) must be given")
    rng = RNGManager(seed=seed)
    tree_conf = PerfectTreeConfig(d=d, height=height)
    markov_height = min(markov_height, height) if markov_height is not None else height
    markov_kwargs = dict(batch_height=markov_height, num_trees=samples, seed=rng.global_numpy_rng)
    echo(f"Simulating with seed: {rng.seed}")
    echo(f"{d}-ary tree with height {height} and batch height {markov_height}")

    if rho is not None:
        echo(f"Ising experiment with rho: {rho}")
        policy = IsingBroadcastPolicy(rho, seed=rng.global_numpy_rng)
        magnet = np.zeros(samples)
        for forest in markov_forest(tree_conf, policy, **markov_kwargs):
            magnet += np.sum(forest.values[-1], axis=1) / math.sqrt(d ** height)
        var, kurtosis = compute_moments(magnet)
        echo(f"Variance: {var}")
        echo(f"Kurtosis: {kurtosis}")
    else:
        echo(f"Coloring experiment with k: {k}")
        policy = ColoringBroadcastPolicy(k, seed=rng.global_numpy_rng)
        free_count = 0
        constrained_count = 0
        unsat_total_count = 0
        unsat_count = {i: 0 for i in range(height - markov_height)}
        leaves = np.empty((samples, d ** height))
        batch_leaves = d ** markov_height
        for j, forest in enumerate(markov_forest(tree_conf, policy, **markov_kwargs)):
            leaves[:, j * batch_leaves:(j + 1) * batch_leaves] = forest.values[-1]
        for i in range(samples):
            tree = InferenceTree(tree_conf, leaves[i])
            try:
                constraint = get_root_constraint(tree, k)
                if constraint is None:
                    free_count += 1
                else:
                    constrained_count += 1
            except UnsatisfiedException as e:
                unsat_total_count += 1
                unsat_count[e.depth] += 1
        echo(f"Free trees: {free_count}")
        echo(f"Constrained trees: {constrained_count}")
        echo(f"Unsatisfied trees: {unsat_total_count}")
        if unsat_total_count > 0:
            for i in range(height - markov_height):
                echo(f"  at depth {i}: {unsat_count[i]}")


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
    return (d ** (height - 1)) * (d + 1) - 1


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


def display_recon_stat(stat):
    echo(f"Unsatisfied: {stat['unsatisfied']['total']}")
    for depth in range(len(stat["unsatisfied"]["details"])):
        echo(f"  At depth {depth}: {stat['unsatisfied']['details'][depth]}")
    echo(f"Invalid: {stat['invalid']}")
    echo(f"Constrained: {stat['constrained']}")
    echo(f"Free: {stat['free']}")


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


@main_process
def sample_validate(sample_every, sample_max_tokens, engine, prompt, step, num_iterations, model, ctx):
    if sample_every is not None and (step % sample_every == 0 or step == num_iterations):
        model.eval()
        echo("Plain sample:")
        tree = engine.generate_tree(prompt, sample_max_tokens)[0]
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


def make_prompt(tokenizer: PerfectTreeTokenizer, config: PerfectTreeConfig):
    if isinstance(tokenizer, SummaryTokenizer):
        return tokenizer.init_summary_tokens(config)
    return [tokenizer.bos_token]


def get_engine(tokenizer: PerfectTreeTokenizer, sampler):
    if isinstance(tokenizer, SummaryTokenizer):
        return StatefulEngine(tokenizer, sampler)
    return SimpleEngine(tokenizer, sampler)


def get_domain(k):
    if k is None:
        return IsingDomain()
    return ColoringDomain(k)


def get_policy(rho, k, rng):
    if rho is not None:
        return IsingBroadcastPolicy(rho, seed=rng), {
            "policy": "Ising",
            "rho": rho,
        }
    return ColoringBroadcastPolicy(k, seed=rng), {
        "policy": "coloring",
        "k": k,
    }


def get_tokenizer(summary_mode, vocab_size, domain):
    if summary_mode == "segment":
        return SegmentSummaryTokenizer(vocab_size, domain)
    elif summary_mode == "path":
        return PathSummaryTokenizer(vocab_size, domain)
    return PerfectTreeTokenizer(vocab_size, domain)


def get_dataloader(config: PerfectTreeConfig, policy, device_batch_size, context_len, batch_height, tokenizer,
                   data_mode="stream", summary=False, device="cpu", seed=None):
    dataloader_kwargs = dict(tokenizer=tokenizer, config=config, policy=policy, batch_size=device_batch_size,
                             seq_len=context_len, batch_height=batch_height, mode=data_mode, device=device,
                             seed=seed)
    return broadcast_tree_data_loader(**dataloader_kwargs, summary=summary)


cli()
