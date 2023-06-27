import torch


def diff(u, x):
    grad = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]
    return grad
