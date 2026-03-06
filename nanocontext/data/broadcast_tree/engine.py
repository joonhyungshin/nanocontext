import torch

from .tokenizer import PerfectTreeTokenizer


class Engine:
    def __init__(self, tokenizer: PerfectTreeTokenizer, sampler):
        self.tokenizer = tokenizer
        self.sampler = sampler

    @property
    def model(self):
        return self.sampler.model

    @property
    def device(self):
        return self.sampler.device

    def generate_tree_tokens_tensor(self, prompt, max_tokens, num_samples=1, allow_many=False, **kwargs):
        raise NotImplementedError

    def generate_tree_tokens(self, prompt, num_samples=1, max_tokens=None, **kwargs) -> list:
        raise NotImplementedError

    def generate_tree(self, prompt, num_samples=1, max_tokens=None, **kwargs):
        tree_tokens = self.generate_tree_tokens(prompt, num_samples=num_samples, max_tokens=max_tokens, **kwargs)
        trees = [next(self.tokenizer.decode_trees_stream(tree_token)) for tree_token in tree_tokens]
        return trees


class SimpleEngine(Engine):
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

    def generate_tree_tokens_tensor_stream(self, prompt, num_samples=1, allow_many=False, max_states=None, **kwargs):
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

    def generate_tree_tokens_tensor(self, prompt, max_tokens, num_samples=1, allow_many=False, **kwargs):
        summary_len, content_len = self.get_summary_and_context_len(prompt)
        result = torch.full((num_samples, max_tokens), self.tokenizer.bos_token,
                            dtype=torch.long, device=self.device)
        max_states = (max_tokens + content_len - 1) // content_len
        num_tokens = 0
        for tokens_tensor in self.generate_tree_tokens_tensor_stream(prompt, num_samples=num_samples,
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
        for content_tokens in self.generate_tree_tokens_tensor_stream(prompt, num_samples=num_samples, **kwargs):
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
