import numpy as np

from .broadcast import BroadcastPolicy
from . import ValueDomain


class ColoringDomain(ValueDomain):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def get_size(self):
        return self.k

    def value_to_index(self, value):
        return value

    def index_to_value(self, index):
        return index % self.k

    def value_to_char(self, value):
        return chr(ord('A') + value)


class ColoringBroadcastPolicy(BroadcastPolicy):
    def __init__(self, k, seed=None):
        self.k = k
        self.domain = ColoringDomain(k)
        self.rng = np.random.default_rng(seed)

    def get_domain(self):
        return self.domain

    def broadcast(self, values=None, multi=1):
        if values is None:
            return self.rng.integers(self.k, size=(multi,))
        values = np.array(values)
        output_shape = (multi,) + values.shape
        shift = self.rng.integers(1, self.k, size=output_shape)
        return (values + shift) % self.k
