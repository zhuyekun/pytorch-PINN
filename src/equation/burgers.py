import numpy as np
import torch

from .base import ConstraintBase
from .derivative import diff


class Burgers1D(ConstraintBase):
    def __init__(self, nu=0.01 / torch.pi, name="Burgers1D"):
        self.name = name

        if not isinstance(nu, torch.Tensor):
            self.nu = torch.tensor(nu)
        else:
            self.nu = nu

    def residual(self, u, t, x):
        u_t = diff(u, t)
        u_x = diff(u, x)
        # print(u)
        u_xx = diff(u_x, x)
        # print((u_t + u * u_x - self.nu * u_xx).shape)
        return u_t + u * u_x - self.nu * u_xx


class InitialCondition(ConstraintBase):
    def __init__(self, name="InitialCondition"):
        self.name = name

    def residual(self, u, t, x):
        return u - torch.sin(torch.pi * x)


class BoundaryCondition(ConstraintBase):
    def __init__(self, name="BoundaryCondition"):
        self.name = name

    def residual(self, u, t, x):
        return u - 0
