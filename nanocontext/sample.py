import torch
import torch.nn.functional as F

from .models.attention import KVCache
from .utils import autocast, get_torch_rng


class NanochatSampler:
    def __init__(self, model, min_context_len=None, max_context_len=None, seed=None):
        self.model = model
        self.min_context_len = min_context_len or self.model.config.sequence_len
        self.max_context_len = max_context_len or self.model.config.rotary_seq_len
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
    def stream(self, tokens, num_samples=1, temperature=1.0, top_k=None):
        if isinstance(tokens, torch.Tensor):
            x = tokens
        else:
            x = torch.tensor([tokens], dtype=torch.long, device=self.device)
        kv_cache = KVCache(
            batch_size=num_samples, seq_len=self.model.config.sequence_len,
            n_heads=self.model.config.n_heads,
            head_dim=self.model.config.n_embd // self.model.config.n_heads,
            n_layers=self.model.config.n_layers
        )
        logits = self._prefill_context_and_forward(x, kv_cache=kv_cache)
        logits = logits[:, -1, :].expand(num_samples, -1)
        context_window = torch.empty((num_samples, self.min_context_len), dtype=torch.long, device=self.device)
        context_window_pos = 0
        context_shift = self.max_context_len - self.min_context_len
        while True:
            next_x = self.sample_next_token(logits, temperature=temperature, top_k=top_k)
            next_tokens = next_x[:, 0]
            yield next_tokens

            if kv_cache.get_pos() > context_shift:
                context_window[:, context_window_pos] = next_x[:, 0]
                context_window_pos += 1
            if context_window_pos >= self.min_context_len:
                logits = self._prefill_context_and_forward(context_window, kv_cache=kv_cache)
                logits = logits[:, -1, :]
                context_window_pos -= min(self.min_context_len, context_shift + 1)
                if context_window_pos > 0:
                    context_window[:, :context_window_pos] = context_window[:, -context_window_pos:].clone()
            else:
                x = next_tokens.unsqueeze(1)
                with autocast():
                    logits = self.model(x, kv_cache=kv_cache)[:, -1, :]

    def generate_tensor(self, tokens, max_tokens=None, num_samples=1, max_context_tokens=None, end_token=None,
                        context_start_token=None, context_end_token=None, always_start=True, pad_token=None,
                        **kwargs):
        if isinstance(tokens, torch.Tensor) and not always_start:
            completed = [end_token is not None and token == end_token for token in tokens[:, -1]]
        else:
            completed = [not always_start and tokens[-1] == end_token] * num_samples

        if pad_token is None:
            pad_token = 0 if end_token is None else end_token
        if all(completed):
            return
        num_generated = torch.zeros(num_samples, dtype=torch.long, device=self.device)
        in_context = -torch.ones(num_samples, dtype=torch.long, device=self.device)
        completed = torch.tensor(completed, dtype=torch.bool, device=self.device)
        max_context_tokens = max_context_tokens or torch.inf
        max_tokens = max_tokens or torch.inf
        for _, next_tokens in enumerate(self.stream(tokens, num_samples=num_samples, **kwargs)):
            next_tokens = next_tokens.clone()
            next_tokens[completed] = pad_token
            yield next_tokens
            num_generated[(in_context == -1) & ~completed & (next_tokens != context_start_token)] += 1
            in_context[(next_tokens == context_end_token) & (in_context >= 0)] = -1
            in_context[((next_tokens == context_start_token) & (in_context == -1))
                       | (in_context >= 0)] += 1
            completed[(num_generated >= max_tokens)
                      | (in_context >= max_context_tokens)
                      | ((next_tokens == end_token) & (in_context == -1))] = True
            if completed.all():
                break

    def generate(self, *args, **kwargs):
        for tokens in self.generate_tensor(*args, **kwargs):
            yield tokens.tolist()

    def generate_batch_tensor(self, tokens, max_tokens, num_samples=1, max_context_tokens=None, end_token=None,
                              context_start_token=None, context_end_token=None, always_start=True, pad_token=None,
                              **kwargs):
        if pad_token is None:
            pad_token = 0 if end_token is None else end_token
        results = torch.full((num_samples, max_tokens), pad_token,
                             dtype=torch.long, device=self.device)
        num_generated = torch.zeros((num_samples, 1), dtype=torch.long, device=self.device)
        indices = torch.arange(max_tokens, device=self.device)
        in_context = -torch.ones(num_samples, dtype=torch.long, device=self.device)
        generator = self.generate_tensor(tokens, max_tokens=max_tokens, num_samples=num_samples,
                                         max_context_tokens=max_context_tokens, end_token=end_token,
                                         context_start_token=context_start_token, context_end_token=context_end_token,
                                         always_start=always_start, pad_token=pad_token, **kwargs)
        for _, next_tokens in enumerate(generator):
            is_active = (in_context == -1) & (next_tokens != context_start_token) & (next_tokens != pad_token)
            results[(indices == num_generated) & is_active.unsqueeze(1)] = next_tokens[is_active]
            num_generated[is_active.unsqueeze(1)] += 1
            in_context[(next_tokens == context_end_token) & (in_context >= 0)] = -1
            in_context[((next_tokens == context_start_token) & (in_context == -1))
                       | (in_context >= 0)] += 1
        return results

    def generate_batch(self, tokens, max_tokens, num_samples=1, max_context_tokens=None, end_token=None,
                       context_start_token=None, context_end_token=None, always_start=True, **kwargs):
        results = [[] for _ in range(num_samples)]
        completed = [not always_start and tokens[-1] == end_token] * num_samples
        if completed[0]:
            return results
        in_context = [-1] * num_samples  # -1: normal / >=0: summary mode
        for next_tokens in self.generate(tokens, num_samples=num_samples, **kwargs):
            for i, next_token in enumerate(next_tokens):
                if completed[i]:
                    continue
                if in_context[i] == -1:
                    if next_token == context_start_token:
                        in_context[i] = 0
                    else:
                        results[i].append(next_token)
                        if len(results[i]) >= max_tokens or next_token == end_token:
                            completed[i] = True
                else:
                    if next_token == context_end_token:
                        in_context[i] = -1
                    else:
                        in_context[i] += 1
                        if max_context_tokens is not None and in_context[i] >= max_context_tokens:
                            # force complete the row
                            completed[i] = True

            if all(completed):
                break
        return results
