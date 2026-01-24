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
