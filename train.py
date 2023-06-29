import datetime
import os
import shutil

import torch

from src.config import get_config
from src.dataset import DataGenerator1D
from src.equation import BoundaryCondition, Burgers1D, InitialCondition
from src.models import Mlp


def set_requires_grad(tensor_dict, requires_grad=True, device="cuda"):
    """
    Set the requires_grad attribute of all tensors in a nested dictionary.

    Args:
        tensor_dict (dict): A nested dictionary containing tensors.
        requires_grad (bool): Whether to require gradient computation or not.
        device (str): The device to move tensors to.
    """
    _tensor_dict = {}
    for k, v in tensor_dict.items():
        if isinstance(v, dict):
            _tensor_dict[k] = set_requires_grad(v, requires_grad, device=device)
        else:
            _tensor_dict[k] = v.clone().detach().requires_grad_(True).to(device)
            # _tensor_dict[k] = torch.tensor(
            #     v, device=device, requires_grad=requires_grad
            # )
    del tensor_dict
    return _tensor_dict


def train(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data pipeline
    cfg_data = config.data_config
    dataset = DataGenerator1D(cfg_data.sample_config)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
    )

    # Constraints and equation
    bc = BoundaryCondition()
    ic = InitialCondition()
    eq = Burgers1D(cfg_data.nu)

    # Model
    cfg_model = config.model_config
    model = Mlp(cfg_model.units, activation=torch.nn.Tanh)
    model = model.to(device)
    model.train()

    def u_sol(t, x):
        u = model(torch.cat([t[:, None], x[:, None]], dim=-1))
        return u.squeeze()

    loss_weights = config.train_config.loss_weight

    def loss(fn, inputs):
        u = {k: fn(**v) for k, v in inputs.items()}
        loss_dict = {
            "boundary": bc.loss(u["boundary"], **inputs["boundary"]),
            "initial": ic.loss(u["initial"], **inputs["initial"]),
            "interior": eq.loss(u["interior"], **inputs["interior"]),
        }
        total_loss = sum([loss_dict[k] * loss_weights[k] for k in loss_dict.keys()])

        return total_loss, loss_dict

    # Optimizer
    cfg_train = config.train_config
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg_train.lr,
        weight_decay=cfg_train.weight_decay,
    )

    loader = iter(dataloader)

    print("Start training...")
    for i in range(cfg_train.epochs):
        inputs = set_requires_grad(next(loader))

        optimizer.zero_grad()
        _loss, loss_dict = loss(u_sol, inputs)
        _loss.backward()
        optimizer.step()

        if i % cfg_train.log_interval == 0:
            _loss_dict = {k: v.item() for k, v in loss_dict.items()}
            print(f"Step {i}, total loss {_loss.item():.4f}, loss_dict {_loss_dict}")

    print("Training finished!")
    save_model(model, cfg_train.save_dir)


def save_model(model, path="/workspaces/pytorch-PINN/results"):
    if not os.path.exists(path):
        os.makedirs(path)

    now = datetime.datetime.now()
    timestr = now.strftime("%Y-%m-%d_%H:%M:%S")

    save_path = os.path.join(path, timestr)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        print("The directory already exist. Overwriting...")

    torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))

    # copy config file
    shutil.copy(
        "/workspaces/pytorch-PINN/src/config.py", os.path.join(save_path, "config.py")
    )

    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    config = get_config()
    train(config)
