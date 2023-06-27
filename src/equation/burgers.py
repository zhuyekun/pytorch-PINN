import numpy as np

from .base import EquationBase
from .derivative import diff


class Burgers1D(EquationBase):
    def __init__(self, nu=0.01 / np.pi, name="Burgers1D"):
        super().__init__(name)

        self.nu = nu

    def residual(self, u, t, x):
        u_t = diff(u, t)
        u_x = diff(u, x)
        u_xx = diff(u_x, x)
        return u_t + u * u_x - self.nu * u_xx
