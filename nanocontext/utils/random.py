import numpy as np
import torch

from .dist import ddp_rank


def get_numpy_rng(seed=None, local=True):
    if isinstance(seed, RNGManager):
        return seed.numpy_rng(local=local)
    return np.random.default_rng(seed=seed)


def get_torch_rng(device="cpu", seed=None, local=True):
    if isinstance(seed, torch.Generator):
        return seed
    elif isinstance(seed, RNGManager):
        return seed.torch_rng(device=device, local=local)
    rng = torch.Generator(device=device)
    if seed is not None:
        rng.manual_seed(seed)
    return rng


class RNGManager:
    def __init__(self, seed=None):
        global_sq = np.random.SeedSequence(seed)
        self.seed = global_sq.entropy
        self.global_numpy_rng = np.random.default_rng(seed=global_sq)
        self.global_torch_seed = global_sq.spawn(1)[0].generate_state(1)[0].item()
        self._global_torch_rng_dict = {}
        self._local_sq = None
        self._local_numpy_rng = None
        self._local_torch_seed = None
        self._local_torch_rng_dict = {}

    def _get_local_sq(self):
        if self._local_sq is None:
            self._local_sq = np.random.SeedSequence([ddp_rank(), self.seed])
        return self._local_sq

    def numpy_rng(self, local=True):
        return self.local_numpy_rng if local else self.global_numpy_rng

    def torch_rng(self, device="cpu", local=True):
        device = torch.device(device)
        torch_rng_dict = self._local_torch_rng_dict if local else self._global_torch_rng_dict
        torch_seed = self.local_torch_seed if local else self.global_torch_seed
        if device not in torch_rng_dict:
            rng = torch.Generator(device=device)
            rng.manual_seed(torch_seed)
            torch_rng_dict[device] = rng
        return torch_rng_dict[device]

    def global_torch_rng(self, device="cpu"):
        return self.torch_rng(device, local=False)

    @property
    def local_numpy_rng(self):
        if self._local_numpy_rng is None:
            self._local_numpy_rng = np.random.default_rng(seed=self._get_local_sq())
        return self._local_numpy_rng

    @property
    def local_torch_seed(self):
        if self._local_torch_seed is None:
            self._local_torch_seed = self._get_local_sq().spawn(1)[0].generate_state(1)[0].item()
        return self._local_torch_seed

    def local_torch_rng(self, device="cpu"):
        return self.torch_rng(device, local=True)
