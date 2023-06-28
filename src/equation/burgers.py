import numpy as np
import torch
from torch.autograd import grad

from .base import ConstraintBase


class Burgers1D(ConstraintBase):
    def __init__(self, nu=0.01 / torch.pi, loss_weight=1, name="Burgers1D"):
        self.name = name
        self.loss_weight = loss_weight

        if not isinstance(nu, torch.Tensor):
            self.nu = torch.tensor(nu)
        else:
            self.nu = nu

    def residual(self, u, inputs):
        derivatives = grad(
            u["interior"],
            inputs["interior"],
            grad_outputs=torch.ones_like(u),
            create_graph=True,
        )
        u_t = derivatives[:, 1]
        u_x = derivatives[:, 2]
        print(u, u_x)
        u_xx = diff(u_x, x)
        return u_t + u["interior"] * u_x - self.nu * u_xx


class InitialCondition(ConstraintBase):
    def __init__(self, loss_weight=1, name="InitialCondition"):
        self.name = name
        self.loss_weight = loss_weight

    def residual(self, u, inputs):
        x = inputs["initial"][:, 1]
        return u["initial"] - torch.sin(torch.pi * x)


class BoundaryCondition(ConstraintBase):
    def __init__(self, loss_weight=1, name="BoundaryCondition"):
        self.name = name
        self.loss_weight = loss_weight

    def residual(self, u, inputs):
        return u["boundary"] - 0
