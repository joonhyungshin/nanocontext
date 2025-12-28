"""
Adapted from nanochat.
"""
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.optim import AdamW

from nanocontext.optim import DistAdamW, Muon, DistMuon
from nanocontext.utils import autocast, echo


@dataclass
class NanochatTrainerConfig:
    # Optimizer config
    unembedding_lr: float = 0.004
    embedding_lr: float = 0.2
    matrix_lr: float = 0.02
    weight_decay: float = 0.0
    # grad_clip: float = 1.0

    # LR scheduler config
    warmup_ratio: float = 0.0
    warmdown_ratio: float = 0.2
    final_lr_frac: float = 0.0


class NanochatTrainer:
    def __init__(self, config, model, dataloader):
        self.config = config
        self.model = model
        self.compiled_model = torch.compile(model, dynamic=False)
        self.dataloader = dataloader
        self.optimizers = self.get_optimizers()

    def train(self, num_iterations, grad_accum_steps):
        step = 0
        train_loss = 0
        x, y = next(self.dataloader)
        while True:
            last_step = step == num_iterations
            if last_step:
                break
            for micro_step in range(grad_accum_steps):
                with autocast():
                    loss = self.compiled_model(x, y)
                train_loss = loss.detach()
                loss = loss / grad_accum_steps
                loss.backward()
                x, y = next(self.dataloader)
            # if self.config.grad_clip > 0.0:
            #     grad_norm_tensor = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            #     grad_norm = grad_norm_tensor.item()
            lrm = self.get_lr_multiplier(step, num_iterations)
            muon_momentum = self.get_muon_momentum(step)
            for opt in self.optimizers:
                for group in opt.param_groups:
                    group["lr"] = group["initial_lr"] * lrm
                if isinstance(opt, Muon) or isinstance(opt, DistMuon):
                    for group in opt.param_groups:
                        group["momentum"] = muon_momentum
            for opt in self.optimizers:
                opt.step()
            self.compiled_model.zero_grad(set_to_none=True)
            step += 1
            echo(f"Step {step} done")

    def get_optimizers(self):
        model_dim = self.model.config.n_embd
        matrix_params = list(self.model.transformer.h.parameters())
        embedding_params = list(self.model.transformer.wte.parameters())
        lm_head_params = list(self.model.lm_head.parameters())
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        adam_groups = [
            dict(params=lm_head_params, lr=self.config.unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=self.config.embedding_lr * dmodel_lr_scale),
        ]
        adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=self.config.weight_decay)
        muon_kwargs = dict(lr=self.config.matrix_lr, momentum=0.95)
        if dist.is_torchelastic_launched():
            adamw_optimizer = DistAdamW(adam_groups, **adamw_kwargs)
            muon_optimizer = DistMuon(matrix_params, **muon_kwargs)
        else:
            adamw_optimizer = AdamW(adam_groups, fused=True, **adamw_kwargs)
            muon_optimizer = Muon(matrix_params, **muon_kwargs)
        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return optimizers

    def get_lr_multiplier(self, step, num_iterations):
        warmup_iters = round(self.config.warmup_ratio * num_iterations)
        warmdown_iters = round(self.config.warmdown_ratio * num_iterations)
        if step < warmup_iters:
            return (step + 1) / warmup_iters
        elif step <= num_iterations - warmdown_iters:
            return 1.0
        else:
            progress = (num_iterations - step) / warmdown_iters
            return progress * 1.0 + (1 - progress) * self.config.final_lr_frac

    @staticmethod
    def get_muon_momentum(step):
        frac = min(step / 300, 1)
        momentum = (1 - frac) * 0.85 + frac * 0.95
        return momentum
