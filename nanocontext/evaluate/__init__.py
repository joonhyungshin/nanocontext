from nanocontext.data.broadcast_tree import Engine, SummaryTokenizer
from nanocontext.tree import PerfectTreeConfig


def infer_summary_every(engine: Engine, prompt, tree_config: PerfectTreeConfig):
    tokenizer = engine.tokenizer
    sampler = engine.sampler
    num_tokens = 0
    d, height = tree_config.d, tree_config.height
    max_tokens = d ** (height - 1) * (d + 1) - 1
    if not isinstance(tokenizer, SummaryTokenizer):
        return None
    summary_start_token = tokenizer.summary_start_token
    for token, _ in sampler.stream(prompt):
        if token == summary_start_token:
            return num_tokens
        else:
            num_tokens += 1
            if num_tokens >= max_tokens:
                break
    return None
