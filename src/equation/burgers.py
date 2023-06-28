import torch

from .base import ConstraintBase
from .derivative import diff


class Burgers1D(ConstraintBase):
    def __init__(self, nu=0.01 / torch.pi, name="Burgers1D"):
        self.name = name
        self.nu = nu

    def residual(self, u, t, x):
        u_t = diff(u["interior"], t)
        u_x = diff(u["interior"], x)
        u_xx = diff(u_x, x)
        return u_t + u["interior"] * u_x - self.nu * u_xx


class InitialCondition(ConstraintBase):
    def __init__(self, name="InitialCondition"):
        self.name = name

    def residual(self, u, t, x):
        return u["initial"] - torch.sin(torch.pi * x)


class BoundaryCondition(ConstraintBase):
    def __init__(self, name="BoundaryCondition"):
        self.name = name

    def residual(self, u, t, x):
        return u["boundary"] - 0
