import numpy as np
import torch

from .compute import device_to_use
from .dist import ddp_rank


def get_seeds(seed=None):
    sq = np.random.SeedSequence(seed)
    return sq.entropy, sq.spawn(1)[0].generate_state(1)[0].item()


class RNGManager:
    def __init__(self, seed=None):
        global_sq = np.random.SeedSequence(seed)
        self.seed = global_sq.entropy
        self.global_numpy_rng = np.random.default_rng(seed=global_sq)
        self._global_torch_seed = global_sq.spawn(1)[0].generate_state(1)[0].item()
        self._local_sq = None
        self._local_numpy_rng = None
        self._global_torch_rng = None
        self._local_torch_rng = None

    @property
    def global_torch_rng(self):
        if self._global_torch_rng is None:
            self._global_torch_rng = torch.Generator(device=device_to_use())
            self._global_torch_rng.manual_seed(self._global_torch_seed)
        return self._global_torch_rng

    def _get_local_sq(self):
        if self._local_sq is None:
            self._local_sq = np.random.SeedSequence([ddp_rank(), self.seed])
        return self._local_sq

    @property
    def local_numpy_rng(self):
        if self._local_numpy_rng is None:
            self._local_numpy_rng = np.random.default_rng(seed=self._get_local_sq())
        return self._local_numpy_rng

    @property
    def local_torch_rng(self):
        if self._local_torch_rng is None:
            local_torch_seed = self._get_local_sq().spawn(1)[0].generate_state(1)[0].item()
            self._local_torch_rng = torch.Generator(device=device_to_use())
            self._local_torch_rng.manual_seed(local_torch_seed)
        return self._local_torch_rng
