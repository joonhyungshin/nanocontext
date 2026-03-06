import numpy as np

from .broadcast import BroadcastPolicy
from . import ValueDomain


class IsingDomain(ValueDomain):
    def get_size(self):
        return 2

    def value_to_index(self, value):
        return 0 if value < 0 else 1

    def index_to_value(self, index):
        return -1 if index == 0 else 1

    def value_to_char(self, value):
        return "-" if value < 0 else "+"


class IsingBroadcastPolicy(BroadcastPolicy):
    def __init__(self, rho, seed=None):
        self.rho = rho
        self.domain = IsingDomain()
        self.rng = np.random.default_rng(seed)

    def get_domain(self):
        return self.domain

    @property
    def flip_prob(self):
        return [(1 - self.rho) / 2, (1 + self.rho) / 2]

    def broadcast(self, values=None, multi=1):
        if values is None:
            return self.rng.choice([-1, 1], size=(multi,))
        values = np.array(values)
        output_shape = (multi,) + values.shape
        flip = self.rng.choice([-1, 1], size=output_shape, p=self.flip_prob)
        return flip * values
