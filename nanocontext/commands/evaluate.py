import click

from nanocontext.data.broadcast_tree import StatefulEngine, load_engine
from nanocontext.evaluate.coloring import check_validity, evaluate_entropy
from nanocontext.evaluate.ising import evaluate_moments, evaluate_perplexity
from nanocontext.tree import IsingBroadcastChannel, PerfectTreeConfig
from nanocontext.tree.coloring import ColoringSpace
from nanocontext.tree.ising import IsingSpace
from nanocontext.utils import  ddp_context, device_to_use, RNGManager

from .common import echo, make_prompt, get_max_tokens, display_recon_stat


@click.command()
@click.option("-d", help="number of children of a tree", type=int, required=True)
@click.option("--height", help="height of a tree", type=int, required=True)
@click.option("--eval-height", help="height to use in evaluation", type=int)
@click.option("--entropy", "eval_entropy", help="evaluate entropy", is_flag=True)
@click.option("--rho", help="correlation for Ising experiment", type=float)
@click.option("--samples", help="number of samples to generate", default=1024, type=int)
@click.option("--sample-batch", help="batch size for sampling", type=int)
@click.option("--model-path", help="path to model", type=str, required=True)
@click.option("--batch-height", help="batch height to stream a tree", type=int)
@click.option("--seed", help="random seed", type=int)
def evaluate(d, height, rho, eval_entropy, eval_height, samples, sample_batch, model_path, batch_height,
             seed):
    rng = RNGManager(seed=seed)
    echo(f"evaluating with seed: {rng.seed}")
    echo(f"using model: {model_path}")
    eval_height = min(eval_height, height) if eval_height is not None else height
    tree_conf = PerfectTreeConfig(d=d, height=height)
    eval_tree_conf = PerfectTreeConfig(d=d, height=eval_height)

    with ddp_context():
        device = device_to_use()
        engine = load_engine(model_path, device, seed=rng.local_torch_rng(device))
        engine.model.eval()
        value_space = engine.tokenizer.value_space
        prompt = make_prompt(engine.tokenizer, tree_conf)
        max_tokens = get_max_tokens(d, eval_height)

        echo(f"generating {samples} samples...")
        if isinstance(value_space, IsingSpace):
            echo("detected Ising experiment.")
            var, kurtosis = evaluate_moments(engine, prompt, samples, max_tokens,
                                             batch_samples=sample_batch, actual_tokens_hint=d**eval_height)
            echo(f"Variance: {var}")
            echo(f"Kurtosis: {kurtosis}")
            if eval_entropy:
                if rho is None:
                    raise ValueError("rho must be provided in Ising experiment to evaluate entropy.")
                channel = IsingBroadcastChannel(rho=rho, seed=rng.local_numpy_rng)
                perplexity = evaluate_perplexity(engine, samples, eval_tree_conf, channel,
                                                 batch_samples=sample_batch, batch_height=batch_height)
                echo(f"Cross entropy: {perplexity}")
        elif isinstance(value_space, ColoringSpace):
            echo("detected Coloring experiment.")
            stat = check_validity(engine, prompt, samples, max_tokens, eval_tree_conf,
                                  batch_samples=sample_batch, patch=True)
            display_recon_stat(stat)
            echo(f"Valid rate: {stat['constrained'] + stat['free']} / {samples}")
            if eval_entropy:
                if isinstance(engine, StatefulEngine) and engine.content_len is not None:
                    num_summaries = (max_tokens - 1) // engine.content_len
                    max_tokens += num_summaries * (engine.summary_len or len(prompt))
                entropy = evaluate_entropy(engine, prompt, samples, max_tokens, eval_tree_conf,
                                           batch_samples=sample_batch)
                echo(f"Entropy: {entropy}")
        else:
            raise click.ClickException("Unknown domain type")
