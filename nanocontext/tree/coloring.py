import numpy as np

from .broadcast import BroadcastChannel
from . import StateSpace


class ColoringSpace(StateSpace):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def get_size(self):
        return self.k

    def state_to_index(self, state):
        return state

    def index_to_state(self, index):
        return index % self.k

    def state_to_char(self, state):
        return chr(ord('A') + state)

    def __eq__(self, other):
        if not isinstance(other, ColoringSpace):
            return NotImplemented
        return self.k == other.k


class ColoringBroadcastChannel(BroadcastChannel):
    def __init__(self, k, seed=None):
        self.k = k
        self.state_space = ColoringSpace(k)
        self.rng = np.random.default_rng(seed)

    def get_state_space(self):
        return self.state_space

    def broadcast(self, values=None, multi=1):
        if values is None:
            return self.rng.integers(self.k, size=(multi,))
        values = np.array(values)
        output_shape = (multi,) + values.shape
        shift = self.rng.integers(1, self.k, size=output_shape)
        return (values + shift) % self.k
