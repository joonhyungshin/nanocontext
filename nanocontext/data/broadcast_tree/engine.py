from dataclasses import asdict

import torch

from nanocontext.sample import NanochatSampler
from nanocontext.utils import device_to_use
from nanocontext.models.nanochat import NanochatConfig, Nanochat
from nanocontext.tree.coloring import ColoringDomain
from nanocontext.tree.ising import IsingDomain

from .tokenizer import PerfectTreeTokenizer, SegmentSummaryTokenizer, PathSummaryTokenizer


class Engine:
    def __init__(self, tokenizer: PerfectTreeTokenizer, sampler: NanochatSampler):
        self.tokenizer = tokenizer
        self.sampler = sampler

    @property
    def model(self):
        return self.sampler.model

    @property
    def device(self):
        return self.sampler.device

    def generate_tree_tokens_tensor_stream(self, prompt, num_samples=1, max_tokens=None, allow_many=False, **kwargs):
        raise NotImplementedError

    def generate_tree_tokens_tensor(self, prompt, max_tokens, num_samples=1, allow_many=False, **kwargs):
        raise NotImplementedError

    def generate_tree_tokens(self, prompt, num_samples=1, max_tokens=None, **kwargs) -> list:
        raise NotImplementedError

    def generate_tree(self, prompt, num_samples=1, max_tokens=None, **kwargs):
        tree_tokens = self.generate_tree_tokens(prompt, num_samples=num_samples, max_tokens=max_tokens, **kwargs)
        trees = [next(self.tokenizer.decode_trees_stream(tree_token)) for tree_token in tree_tokens]
        return trees


class SimpleEngine(Engine):
    def generate_tree_tokens_tensor_stream(self, prompt, num_samples=1, allow_many=False, **kwargs):
        end_token = None if allow_many else self.tokenizer.bos_token
        yield from self.sampler.generate_tensor(prompt, num_samples=num_samples, end_token=end_token,
                                                **kwargs)

    def generate_tree_tokens_tensor(self, prompt, max_tokens, num_samples=1, allow_many=False, **kwargs):
        end_token = None if allow_many else self.tokenizer.bos_token
        return self.sampler.generate_batch_tensor(prompt, max_tokens,
                                                  num_samples=num_samples, end_token=end_token, **kwargs)

    def generate_tree_tokens(self, prompt, num_samples=1, **kwargs):
        return self.sampler.generate_batch(prompt, num_samples=num_samples, end_token=self.tokenizer.bos_token,
                                           **kwargs)


class StatefulEngine(Engine):
    def get_summary_and_context_len(self, prompt):
        summary_len = len(prompt)
        content_len = self.sampler.context_len + 1 - 2 * summary_len
        return summary_len, content_len

    def generate_tokens_tensor_batch_stream(self, prompt, num_samples=1, allow_many=False, max_states=None, **kwargs):
        summary_len, content_len = self.get_summary_and_context_len(prompt)
        max_tokens = content_len + summary_len
        end_token = None if allow_many else self.tokenizer.bos_token
        num_states = 0
        beginning = True
        while True:
            tokens_tensor = self.sampler.generate_batch_tensor(prompt, max_tokens,
                                                               num_samples=num_samples,
                                                               end_token=end_token,
                                                               always_start=beginning, **kwargs)
            yield tokens_tensor[:, summary_len:max_tokens]
            num_states += 1
            if not allow_many and (tokens_tensor[:, max_tokens - 1] == end_token).all():
                break
            if max_states is not None and num_states >= max_states:
                break
            prompt = tokens_tensor[:, max_tokens:]
            beginning = False

    def generate_tree_tokens_tensor_stream(self, prompt, num_samples=1, max_tokens=None, allow_many=False, **kwargs):
        summary_len, content_len = self.get_summary_and_context_len(prompt)
        max_states = (max_tokens + content_len - 1) // content_len
        num_tokens = 0
        for tokens_tensor in self.generate_tokens_tensor_batch_stream(prompt,
                                                                      num_samples=num_samples,
                                                                      max_states=max_states,
                                                                      allow_many=allow_many, **kwargs):
            _, num_batch_tokens = tokens_tensor.shape
            for i in range(num_batch_tokens):
                yield tokens_tensor[:, i]
                num_tokens += 1
                if max_tokens is not None and num_tokens >= max_tokens:
                    break

    def generate_tree_tokens_tensor(self, prompt, max_tokens, num_samples=1, allow_many=False, **kwargs):
        summary_len, content_len = self.get_summary_and_context_len(prompt)
        result = torch.full((num_samples, max_tokens), self.tokenizer.bos_token,
                            dtype=torch.long, device=self.device)
        max_states = (max_tokens + content_len - 1) // content_len
        num_tokens = 0
        for tokens_tensor in self.generate_tokens_tensor_batch_stream(prompt, num_samples=num_samples,
                                                                      allow_many=allow_many, max_states=max_states,
                                                                      **kwargs):
            _, token_len = tokens_tensor.shape
            actual_token_len = min(token_len, max_tokens - num_tokens)
            result[:, num_tokens:num_tokens + actual_token_len] = tokens_tensor[:, :actual_token_len]
            num_tokens += actual_token_len
            if num_tokens >= max_tokens:
                break
        return result

    def generate_tree_tokens(self, prompt, num_samples=1, max_tokens=None, **kwargs):
        tree_tokens = [[] for _ in range(num_samples)]
        completed = [False for _ in range(num_samples)]
        for content_tokens in self.generate_tokens_tensor_batch_stream(prompt, num_samples=num_samples, **kwargs):
            if all(completed):
                break
            for i in range(num_samples):
                for token_raw in content_tokens[i]:
                    token = token_raw.item()
                    if token == self.tokenizer.bos_token:
                        completed[i] = True
                        break
                    if max_tokens is not None and len(tree_tokens[i]) >= max_tokens:
                        completed[i] = True
                        break
                    tree_tokens[i].append(token)
        return tree_tokens


def save_engine(engine: Engine, filename):
    tokenizer = engine.tokenizer
    domain = tokenizer.domain
    model = engine.model
    context_len = engine.sampler.context_len
    if isinstance(engine, SimpleEngine):
        summary = "disabled"
    elif isinstance(tokenizer, SegmentSummaryTokenizer):
        summary = "segment"
    elif isinstance(tokenizer, PathSummaryTokenizer):
        summary = "path"
    else:
        raise ValueError("cannot save engine: unknown engine type")
    if isinstance(domain, IsingDomain):
        domain = {
            "type": "ising"
        }
    elif isinstance(domain, ColoringDomain):
        domain = {
            "type": "coloring",
            "k": domain.k,
        }
    else:
        raise ValueError("cannot save engine: unknown domain")
    state_dict = {
        "summary": summary,
        "max_vocab_size": tokenizer.max_vocab_size,
        "context_len": context_len,
        "domain": domain,
        "model": {
            "type": "nanochat",
            "config": asdict(model.config),
            "parameters": model.state_dict()
        }
    }
    torch.save(state_dict, filename)


def load_engine(filename, device=None, seed=None):
    device = device or device_to_use()
    state_dict = torch.load(filename, map_location=device)
    context_len = state_dict["context_len"]
    summary = state_dict["summary"]
    domain_dict = state_dict["domain"]
    max_vocab_size = state_dict["max_vocab_size"]
    model_state_dict = state_dict["model"]
    model_type = model_state_dict["type"]
    model_config = model_state_dict["config"]
    if domain_dict["type"] == "ising":
        domain = IsingDomain()
    elif domain_dict["type"] == "coloring":
        domain = ColoringDomain(domain_dict["k"])
    else:
        raise ValueError("unknown domain")
    if summary == "disabled":
        tokenizer = PerfectTreeTokenizer(max_vocab_size, domain)
        engine_class = SimpleEngine
    elif summary == "segment":
        tokenizer = SegmentSummaryTokenizer(max_vocab_size, domain)
        engine_class = StatefulEngine
    elif summary == "path":
        tokenizer = PathSummaryTokenizer(max_vocab_size, domain)
        engine_class = StatefulEngine
    else:
        raise ValueError("unknown summary type")
    if model_type != "nanochat":
        raise ValueError("unknown model type")
    with torch.device("meta"):
        config = NanochatConfig(**model_config)
        model = Nanochat(config)
    model.to_empty(device=device)
    model.load_state_dict(model_state_dict["parameters"], strict=True, assign=True)
    model.preprocess()
    sampler = NanochatSampler(model, context_len, seed=seed)
    engine = engine_class(tokenizer, sampler)
    return engine
