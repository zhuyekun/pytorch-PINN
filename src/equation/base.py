import abc

# import numpy as np
import torch


class ConstraintBase(abc.ABC):
    @abc.abstractmethod
    def residual(self):
        pass

    def loss(self, u, t, x):
        return torch.mean(self.residual(u, t, x) ** 2)
