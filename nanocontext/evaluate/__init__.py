from nanocontext.data.broadcast_tree import Engine
from nanocontext.tree import PerfectTreeConfig


def infer_summary_every(engine: Engine, prompt, tree_config: PerfectTreeConfig):
    tokenizer = engine.tokenizer
    num_tokens = 0
    d, height = tree_config.d, tree_config.height
    max_tokens = d ** (height - 1) * (d + 1) - 1
    summary_start_token = getattr(tokenizer, "summary_start_token", None)
    for token in engine.generate_tree_tokens_stream(prompt, ignore_context=False, allow_many=True):
        if token == summary_start_token:
            return num_tokens
        else:
            num_tokens += 1
            if num_tokens >= max_tokens:
                break
    return None
