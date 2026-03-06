import torch


def tokens_to_data(tokens, batch_size, seq_len, device="cpu", compact=True):
    use_cuda_opt = device == "cuda"
    scratch = torch.tensor(tokens, dtype=torch.long, pin_memory=use_cuda_opt)
    if compact:
        assert len(tokens) == batch_size * seq_len + 1
        inputs_cpu = scratch[:-1]
        targets_cpu = scratch[1:]
        inputs = inputs_cpu.view(batch_size, seq_len).to(device=device, non_blocking=use_cuda_opt)
        targets = targets_cpu.view(batch_size, seq_len).to(device=device, non_blocking=use_cuda_opt)
    else:
        assert len(tokens) == batch_size * (seq_len + 1)
        scratch_view = scratch.view(batch_size, seq_len + 1)
        inputs = scratch_view[:, :-1].to(device=device, non_blocking=use_cuda_opt)
        targets = scratch_view[:, 1:].to(device=device, non_blocking=use_cuda_opt)
    return inputs, targets


class BaseTokenizer:
    bos_token = 0
    variable_token_base = 1
    variable_tokens = []

    def __init__(self, max_vocab_size):
        self.max_vocab_size = max_vocab_size
        self.variable_token_bases = {
            token_name: self.variable_token_base + i
            for i, token_name in enumerate(self.variable_tokens)
        }

    def get_variable_token(self, token_name, shift):
        token = self.variable_token_bases[token_name] + shift * len(self.variable_tokens)
        if token >= self.max_vocab_size:
            raise ValueError("max_vocab_size is too small")
        return token

    def decode_variable_token(self, token):
        total_shift = token - self.variable_token_base
        shift, idx = divmod(total_shift, len(self.variable_tokens))
        return self.variable_tokens[idx], shift

    def get_variable_token_name(self, token):
        return self.decode_variable_token(token)[0]

    def is_variable_token(self, token, token_name):
        token_base = self.variable_token_bases[token_name]
        return (token - token_base) % len(self.variable_tokens) == 0
