"""
Adapted from nanochat.
"""
from dataclasses import dataclass
from enum import Enum
from functools import partial
import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim import AdamW

from nanocontext.optim import DistAdamW, Muon, DistMuon
from nanocontext.utils import autocast


@dataclass
class NanochatTrainerConfig:
    # Optimizer config
    unembedding_lr: float = 0.004
    embedding_lr: float = 0.2
    matrix_lr: float = 0.02
    weight_decay: float = 0.0
    grad_clip: float = 1.0

    # LR scheduler config
    warmup_ratio: float = 0.0
    warmdown_ratio: float = 0.2
    final_lr_frac: float = 0.0


class TrainerSignal(Enum):
    PRE_OPTIM_STEP = 1
    POST_OPTIM_STEP = 2


# TODO: make it stateful?
class NanochatTrainer:
    def __init__(self, config, model, dataloader, seed=None):
        self.config = config
        self.model = model
        self.compiled_model = torch.compile(model, dynamic=False)
        self.dataloader = dataloader
        self.optimizers = self.get_optimizers()
        self.callback_registry = {}
        if isinstance(seed, torch.Generator):
            self.rng = seed
        else:
            self.rng = torch.Generator(device=model.device)
            if seed is not None:
                self.rng.manual_seed(seed)

    def register_callback(self, signal, callback, *args, **kwargs):
        self.callback_registry.setdefault(signal, []).append(partial(callback, *args, **kwargs))

    def fire(self, signal, **payload):
        for callback in self.callback_registry.get(signal, []):
            callback(**payload)

    def train(self, num_iterations, grad_accum_steps, loss_reduction="mean"):
        step = 0
        train_loss = 0
        x, y = next(self.dataloader)
        stats = {}
        while True:
            last_step = step == num_iterations
            self.fire(TrainerSignal.PRE_OPTIM_STEP, step=step, num_iterations=num_iterations, model=self.model)
            if last_step:
                break
            for micro_step in range(grad_accum_steps):
                with autocast():
                    logits = self.compiled_model(x)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1),
                                           ignore_index=-1, reduction=loss_reduction)
                train_loss = loss.detach()
                loss = loss / grad_accum_steps
                loss.backward()
                x, y = next(self.dataloader)
            if self.config.grad_clip > 0.0:
                grad_norm_tensor = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                grad_norm = grad_norm_tensor.item()
                stats["grad_norm"] = grad_norm
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
            stats.update({
                "train_loss": train_loss.item(),
                "lrm": lrm,
            })
            self.fire(TrainerSignal.POST_OPTIM_STEP, step=step, num_iterations=num_iterations, stats=stats)
            step += 1

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

    def init_weights(self):
        self.model.apply(self._init_weights)
        torch.nn.init.zeros_(self.model.lm_head.weight)
        for block in self.model.transformer.h:
            torch.nn.init.zeros_(block.mlp.proj.weight)
            torch.nn.init.zeros_(block.attn.proj.weight)
        self.model.preprocess()

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std, generator=self.rng)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0, generator=self.rng)
