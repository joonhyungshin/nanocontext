import torch
import torch.nn.functional as F

from .models.attention import KVCache
from .utils import autocast, get_torch_rng


class NanochatSampler:
    def __init__(self, model, context_len=None, seed=None):
        self.model = model
        self.context_len = context_len or self.model.config.sequence_len
        self.rng = get_torch_rng(device=self.model.device, seed=seed, local=True)

    @property
    def device(self):
        return self.model.device

    @torch.inference_mode()
    def sample_next_token(self, logits, temperature=1.0, top_k=None):
        if temperature == 0.0:
            return torch.argmax(logits, dim=-1, keepdim=True)
        elif top_k is not None:
            k = min(top_k, logits.size(-1))
            vals, idx = torch.topk(logits, k, dim=-1)
            vals = vals / temperature
            probs = F.softmax(vals, dim=-1)
            choice = torch.multinomial(probs, num_samples=1, generator=self.rng)
            return idx.gather(dim=1, index=choice)
        else:
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1, generator=self.rng)

    def _prefill_context_and_forward(self, x, kv_cache):
        batch_size, seq_len = x.shape
        kv_cache_prefill = KVCache(
            batch_size=batch_size, seq_len=seq_len,
            n_heads=self.model.config.n_heads,
            head_dim=self.model.config.n_embd // self.model.config.n_heads,
            n_layers=self.model.config.n_layers
        )
        with autocast():
            logits = self.model(x, kv_cache=kv_cache_prefill)
        kv_cache.prefill(kv_cache_prefill)
        return logits

    @torch.inference_mode()
    def generate_tensor(self, tokens, num_samples=1, max_tokens=None, end_token=None,
                        temperature=1.0, top_k=None, always_start=True):
        completed = [False] * num_samples
        if isinstance(tokens, torch.Tensor):
            x = tokens
            token_len = tokens.shape[1]
            if not always_start:
                completed = [end_token is not None and token == end_token for token in tokens[:, -1]]
        else:
            x = torch.tensor([tokens], dtype=torch.long, device=self.device)
            token_len = len(tokens)
            if not always_start:
                completed = [tokens[-1] == end_token] * num_samples
        kv_length_hint = (token_len + max_tokens) if max_tokens is not None else self.model.config.sequence_len
        kv_cache = KVCache(
            batch_size=num_samples, seq_len=kv_length_hint,
            n_heads=self.model.config.n_heads,
            head_dim=self.model.config.n_embd // self.model.config.n_heads,
            n_layers=self.model.config.n_layers
        )
        logits = self._prefill_context_and_forward(x, kv_cache=kv_cache)
        logits = logits[:, -1, :].expand(num_samples, -1)
        context_window = torch.empty((num_samples, self.context_len), dtype=torch.long, device=self.device)
        context_window_pos = 0
        num_generated = 0
        while True:
            if max_tokens is not None and num_generated >= max_tokens:
                break
            if all(completed):
                break
            next_x = self.sample_next_token(logits, temperature=temperature, top_k=top_k)
            next_tokens = next_x[:, 0]
            if end_token is not None:
                for i in range(num_samples):
                    if next_tokens[i] == end_token:
                        completed[i] = True
            yield next_tokens
            num_generated += 1

            if kv_cache.get_pos() + self.context_len > self.model.config.rotary_seq_len:
                context_window[:, context_window_pos] = next_x[:, 0]
                context_window_pos += 1
            if context_window_pos >= self.context_len:
                logits = self._prefill_context_and_forward(context_window, kv_cache=kv_cache)
                logits = logits[:, -1, :]
                context_window_pos = 0
            else:
                x = next_tokens.unsqueeze(1)
                with autocast():
                    logits = self.model(x, kv_cache=kv_cache)[:, -1, :]

    def generate(self, *args, **kwargs):
        for tokens in self.generate_tensor(*args, **kwargs):
            yield tokens.tolist()

    def generate_batch_tensor(self, tokens, max_tokens, num_samples=1, end_token=None, **kwargs):
        fill_value = 0 if end_token is None else end_token
        token_len = tokens.shape[1] if isinstance(tokens, torch.Tensor) else len(tokens)
        results = torch.full((num_samples, token_len + max_tokens), fill_value,
                             dtype=torch.long, device=self.device)
        if isinstance(tokens, torch.Tensor):
            results[:, :token_len] = tokens
        else:
            results[:, :token_len] = torch.tensor([tokens], dtype=torch.long, device=self.device)
        for i, next_tokens in enumerate(self.generate_tensor(tokens, num_samples=num_samples, max_tokens=max_tokens,
                                                             end_token=end_token, **kwargs)):
            results[:, i + token_len] = next_tokens
        return results

    def generate_batch(self, tokens, num_samples=1, end_token=None, **kwargs):
        results = [tokens.copy() for _ in range(num_samples)]
        completed = [False] * num_samples
        for next_tokens in self.generate(tokens, num_samples=num_samples, end_token=end_token, **kwargs):
            for i, next_token in enumerate(next_tokens):
                if end_token is not None and next_token == end_token:
                    completed[i] = True
                if not completed[i]:
                    results[i].append(next_token)
        return results


class SummarySampler:
    pass
