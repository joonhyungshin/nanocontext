import torch
import torch.nn.functional as F

from .models.attention import KVCache
from .utils import autocast


class NanochatSampler:
    def __init__(self, model, seed=None):
        self.model = model
        self.rng = torch.Generator(model.device)
        if seed is not None:
            self.rng.manual_seed(seed)

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

    @torch.inference_mode()
    def generate(self, tokens, num_samples=1, max_tokens=None, end_token=None, temperature=1.0, top_k=None):
        kv_cache_kwargs = dict(
            n_heads=self.model.config.n_heads,
            head_dim=self.model.config.n_embd // self.model.config.n_heads,
            n_layers=self.model.config.n_layers,
        )
        kv_cache_prefill = KVCache(batch_size=1, seq_len=len(tokens), **kv_cache_kwargs)
        x = torch.tensor([tokens], dtype=torch.long, device=self.device)
        with autocast():
            logits = self.model(x, kv_cache=kv_cache_prefill)
            logits = logits[:, -1, :].expand(num_samples, -1)
        kv_length_hint = (len(tokens) + max_tokens) if max_tokens is not None else self.model.config.sequence_len
        kv_cache = KVCache(batch_size=num_samples, seq_len=kv_length_hint, **kv_cache_kwargs)
        kv_cache.copy_from(kv_cache_prefill)
        completed = [False] * num_samples
        num_generated = 0
        while True:
            if max_tokens is not None and num_generated >= max_tokens:
                break
            if all(completed):
                break
            next_x = self.sample_next_token(logits, temperature=temperature, top_k=top_k)
            next_tokens = next_x[:, 0].tolist()
            if end_token is not None:
                for i in range(num_samples):
                    if next_tokens[i] == end_token:
                        completed[i] = True
            yield next_tokens
            num_generated += 1
            x = torch.tensor(next_tokens, dtype=torch.long, device=self.device).unsqueeze(1)
            with autocast():
                logits = self.model(x, kv_cache=kv_cache)[:, -1, :]
