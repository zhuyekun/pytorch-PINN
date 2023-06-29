import torch
from torch.autograd import grad


def diff(u, x):
    return grad(
        u, x, grad_outputs=torch.ones_like(u), create_graph=True, allow_unused=True
    )[0]
