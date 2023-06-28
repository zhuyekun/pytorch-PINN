import abc

# import numpy as np
import torch


class ConstraintBase(abc.ABC):
    @abc.abstractmethod
    def residual(self):
        pass

    def loss(self, *args):
        return torch.mean(self.residual(*args) ** 2)
