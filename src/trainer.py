import torch

from .config import config
from .equation import BoundaryCondition, Burgers1D, InitialCondition
from .geometry import Rectangle1D
from .models import Mlp, ResNet


def train():
    config = config()

    # Geometry
    cfg_data = config.data_config
    geom_x = Rectangle1D(cfg_data.x_range[0], cfg_data.x_range[1])
    geom_t = Rectangle1D(cfg_data.x_range[0], cfg_data.x_range[1])
