import numpy as np

from .broadcast import BroadcastChannel
from . import StateSpace


class IsingSpace(StateSpace):
    def get_size(self):
        return 2

    def state_to_index(self, state):
        return 0 if state < 0 else 1

    def index_to_state(self, index):
        return -1 if index == 0 else 1

    def state_to_char(self, state):
        return "-" if state < 0 else "+"

    def __eq__(self, other):
        if not isinstance(other, IsingSpace):
            return NotImplemented
        return True


class IsingBroadcastChannel(BroadcastChannel):
    def __init__(self, rho, seed=None):
        self.rho = rho
        self.state_space = IsingSpace()
        self.rng = np.random.default_rng(seed)

    def get_state_space(self):
        return self.state_space

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
