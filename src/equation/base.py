import abc

import numpy as np

# import torch


class EquationBase(abc.ABC):
    @abc.abstractmethod
    def residual(self):
        pass

    def pinn_loss(self, x, y):
        return np.mean(self.residual(x, y) ** 2)
