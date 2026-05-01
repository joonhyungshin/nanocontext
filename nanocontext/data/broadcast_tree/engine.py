from dataclasses import asdict
from typing import Generator

import torch

from nanocontext.sample import NanochatSampler
from nanocontext.utils import device_to_use, d_order
from nanocontext.models.nanochat import NanochatConfig, Nanochat
from nanocontext.tree import PerfectTreeConfig
from nanocontext.tree.coloring import ColoringSpace
from nanocontext.tree.ising import IsingSpace

from .tokenizer import PerfectTreeTokenizer, SegmentSummaryTokenizer, PathSummaryTokenizer, SummaryTokenizer


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

    def generate_tree_tokens_tensor_stream(self, prompt, num_samples=1, max_tokens=None, allow_many=False,
                                           **kwargs) -> Generator[torch.Tensor]:
        raise NotImplementedError

    def generate_tree_tokens_tensor(self, prompt, max_tokens, num_samples=1, allow_many=False,
                                    **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def generate_tree_tokens_stream(self, *args, **kwargs) -> Generator[list]:
        for tensor in self.generate_tree_tokens_tensor_stream(*args, **kwargs):
            yield tensor.tolist()

    def generate_tree_tokens(self, prompt, max_tokens, num_samples=1, allow_many=False, **kwargs) -> list:
        raise NotImplementedError

    def generate_tree(self, prompt, max_tokens, num_samples=1, **kwargs):
        tree_tokens = self.generate_tree_tokens(prompt, max_tokens, num_samples=num_samples, **kwargs)
        trees = [next(self.tokenizer.decode_trees_stream(tree_token)) for tree_token in tree_tokens]
        return trees

    def patch_tree_token(self, tree_token, tree_config: PerfectTreeConfig):
        d, height = tree_config.d, tree_config.height
        bos_fix = 0 if len(tree_token) > 0 and tree_token[0] == self.tokenizer.bos_token else 1
        for idx in range(1, d ** (height - 1)):
            zero_cnt = d_order(idx, d) + 1
            punc_idx = idx * (d + 1) - bos_fix
            if len(tree_token) > punc_idx:
                tree_token[punc_idx] = self.tokenizer.punctuation(zero_cnt)
            else:
                break
        end_idx = (d ** (height - 1)) * (d + 1) - bos_fix
        if len(tree_token) > end_idx:
            tree_token[end_idx] = self.tokenizer.bos_token

    def generate_patched_tree(self, prompt, max_tokens, tree_config: PerfectTreeConfig, num_samples=1, **kwargs):
        tree_tokens = self.generate_tree_tokens(prompt, max_tokens, num_samples=num_samples, **kwargs)
        trees = []
        for tree_token in tree_tokens:
            self.patch_tree_token(tree_token, tree_config)
            trees.append(next(self.tokenizer.decode_trees_stream(tree_token)))
        return trees


class SimpleEngine(Engine):
    def __init__(self, tokenizer, sampler):
        super().__init__(tokenizer, sampler)
        if isinstance(tokenizer, SummaryTokenizer):
            self.context_kwargs = dict(
                context_start_token=tokenizer.summary_start_token,
                context_end_token=tokenizer.summary_end_token,
            )
        else:
            self.context_kwargs = {}

    def generate_tree_tokens_tensor_stream(self, prompt, num_samples=1, allow_many=False,
                                           ignore_context=True, **kwargs):
        end_token = None if allow_many else self.tokenizer.bos_token
        if ignore_context:
            kwargs |= self.context_kwargs
        yield from self.sampler.generate_tensor(prompt, num_samples=num_samples, end_token=end_token,
                                                **kwargs)

    def generate_tree_tokens_tensor(self, prompt, max_tokens, num_samples=1, allow_many=False,
                                    ignore_context=True, **kwargs):
        end_token = None if allow_many else self.tokenizer.bos_token
        if ignore_context:
            kwargs |= self.context_kwargs
        return self.sampler.generate_batch_tensor(prompt, max_tokens,
                                                  num_samples=num_samples, end_token=end_token,
                                                  **kwargs)

    def generate_tree_tokens(self, prompt, max_tokens, num_samples=1, allow_many=False,
                             ignore_context=True, **kwargs):
        end_token = None if allow_many else self.tokenizer.bos_token
        if ignore_context:
            kwargs |= self.context_kwargs
        return self.sampler.generate_batch(prompt, max_tokens, num_samples=num_samples, end_token=end_token,
                                           **kwargs)


class StatefulEngine(Engine):
    def __init__(self, tokenizer, sampler, summary_len=None, content_len=None):
        super().__init__(tokenizer, sampler)
        self.summary_len = summary_len
        self.content_len = content_len

    def generate_tokens_tensor_batch_stream(self, prompt, num_samples=1, allow_many=False, max_states=None,
                                            ignore_context=True, **kwargs):
        summary_len = self.summary_len or len(prompt)
        content_len = self.content_len or self.sampler.min_context_len + 1 - 2 * summary_len
        max_tokens = content_len + summary_len
        end_token = None if allow_many else self.tokenizer.bos_token
        num_states = 0
        beginning = True
        while True:
            tokens_tensor = self.sampler.generate_batch_tensor(prompt, max_tokens,
                                                               num_samples=num_samples,
                                                               end_token=end_token,
                                                               always_start=beginning, **kwargs)
            yield tokens_tensor[:, :content_len] if ignore_context else tokens_tensor
            num_states += 1
            if not allow_many and (tokens_tensor[:, content_len - 1] == end_token).all():
                break
            if max_states is not None and num_states >= max_states:
                break
            prompt = tokens_tensor[:, content_len:]
            beginning = False

    def generate_tree_tokens_tensor_stream(self, prompt, num_samples=1, max_tokens=None,
                                           allow_many=False, **kwargs):
        summary_len = self.summary_len or len(prompt)
        content_len = self.content_len or self.sampler.min_context_len + 1 - 2 * summary_len
        max_states = (max_tokens + content_len - 1) // content_len if max_tokens is not None else None
        num_tokens = 0
        for tokens_tensor in self.generate_tokens_tensor_batch_stream(prompt,
                                                                      num_samples=num_samples,
                                                                      max_states=max_states,
                                                                      allow_many=allow_many,
                                                                      **kwargs):
            _, num_batch_tokens = tokens_tensor.shape
            for i in range(num_batch_tokens):
                yield tokens_tensor[:, i]
                num_tokens += 1
                if max_tokens is not None and num_tokens >= max_tokens:
                    break

    def generate_tree_tokens_tensor(self, prompt, max_tokens, num_samples=1,
                                    allow_many=False, **kwargs):
        summary_len = self.summary_len or len(prompt)
        content_len = self.content_len or self.sampler.min_context_len + 1 - 2 * summary_len
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

    def generate_tree_tokens(self, prompt, max_tokens, num_samples=1, allow_many=False, **kwargs):
        tree_tokens = [[] for _ in range(num_samples)]
        completed = [False for _ in range(num_samples)]
        summary_len = self.summary_len or len(prompt)
        content_len = self.content_len or self.sampler.min_context_len + 1 - 2 * summary_len
        max_states = (max_tokens + content_len - 1) // content_len
        for content_tokens in self.generate_tokens_tensor_batch_stream(prompt, num_samples=num_samples,
                                                                       allow_many=allow_many, max_states=max_states,
                                                                       **kwargs):
            if all(completed):
                break
            for i in range(num_samples):
                for token_raw in content_tokens[i]:
                    if len(tree_tokens[i]) >= max_tokens:
                        break
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
    domain = tokenizer.value_space
    model = engine.model
    min_context_len = engine.sampler.min_context_len
    max_context_len = engine.sampler.max_context_len
    if isinstance(engine, SimpleEngine):
        engine_type = "simple"
        engine_meta = {}
    elif isinstance(engine, StatefulEngine):
        engine_type = "stateful"
        engine_meta = {
            "summary_len": engine.summary_len,
            "content_len": engine.content_len,
        }
    else:
        raise ValueError("cannot save engine: unknown engine type")
    if isinstance(tokenizer, SegmentSummaryTokenizer):
        summary = "segment"
    elif isinstance(tokenizer, PathSummaryTokenizer):
        summary = "path"
    else:
        summary = "disabled"
    if isinstance(domain, IsingSpace):
        domain = {
            "type": "ising"
        }
    elif isinstance(domain, ColoringSpace):
        domain = {
            "type": "coloring",
            "k": domain.k,
        }
    else:
        raise ValueError("cannot save engine: unknown domain")
    state_dict = {
        "engine": {
            "type": engine_type,
            "meta": engine_meta,
        },
        "summary": summary,
        "max_vocab_size": tokenizer.max_vocab_size,
        "min_context_len": min_context_len,
        "max_context_len": max_context_len,
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
    min_context_len = state_dict["min_context_len"]
    max_context_len = state_dict["max_context_len"]
    engine_data = state_dict["engine"]
    engine_type = engine_data["type"]
    engine_meta = engine_data["meta"]
    summary = state_dict["summary"]
    domain_dict = state_dict["domain"]
    max_vocab_size = state_dict["max_vocab_size"]
    model_state_dict = state_dict["model"]
    model_type = model_state_dict["type"]
    model_config = model_state_dict["config"]
    if domain_dict["type"] == "ising":
        domain = IsingSpace()
    elif domain_dict["type"] == "coloring":
        domain = ColoringSpace(domain_dict["k"])
    else:
        raise ValueError("unknown domain")
    if engine_type == "simple":
        engine_class = SimpleEngine
    elif engine_type == "stateful":
        engine_class = StatefulEngine
    else:
        raise ValueError("unknown engine type")
    if summary == "disabled":
        tokenizer = PerfectTreeTokenizer(max_vocab_size, domain)
    elif summary == "segment":
        tokenizer = SegmentSummaryTokenizer(max_vocab_size, domain)
    elif summary == "path":
        tokenizer = PathSummaryTokenizer(max_vocab_size, domain)
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
    sampler = NanochatSampler(model, min_context_len, max_context_len, seed=seed)
    engine = engine_class(tokenizer, sampler, **engine_meta)
    return engine
