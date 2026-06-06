import click

from nanocontext.data.broadcast_tree import load_engine
from nanocontext.tree import PerfectTreeConfig
from nanocontext.utils import device_to_use, RNGManager

from .common import echo, make_prompt


@click.command()
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
    gen_kwargs = dict(num_samples=samples, temperature=temperature, top_k=top_k, max_context_tokens=64)
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
