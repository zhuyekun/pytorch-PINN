import torch

from .config import get_config
from .dataset import DataGenerator1D
from .equation import BoundaryCondition, Burgers1D, InitialCondition
from .geometry import Rectangle1D
from .models import Mlp, ResNet


def train():
    config = get_config()

    # Data pipeline
    cfg_data = config.data_config
    dataset = DataGenerator1D(cfg_data)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        shuffle=True,
        num_workers=0,
    )

    # Constraints and equation
    constraints = []
    constraints.append(BoundaryCondition(config.train_config.loss_weight["boundary"]))
    constraints.append(InitialCondition(config.train_config.loss_weight["initial"]))
    constraints.append(
        Burgers1D(cfg_data.nu, config.train_config.loss_weight["interior"])
    )

    # Model
    cfg_model = config.model_config
    model = Mlp(cfg_model.units)

    # Optimizer
    cfg_train = config.train_config
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg_train.lr,
        weight_decay=cfg_train.weight_decay,
    )

    def loss(model, inputs):
        u = {k: model(v) for k, v in inputs.items()}
        loss_list = [c.loss(u) for c in constraints]
        loss = sum(loss_list)
        return loss

    # for i, inputs in enumerate(dataloader):
