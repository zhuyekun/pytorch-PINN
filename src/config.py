import ml_collections
import numpy as np


def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    config.model_config = ml_collections.ConfigDict()
    config.train_config = ml_collections.ConfigDict()
    config.data_config = ml_collections.ConfigDict()

    # Training configuration
    config.train_config.loss_weight = {"interior": 1.0, "boundary": 1.0, "initial": 1.0}
    config.train_config.epochs = 100
    config.train_config.lr = 1e-3
    config.train_config.weight_decay = 1e-5

    # Model configuration
    config.model_config.units = [2, 128, 128, 2]

    # Data configuration
    config.data_config.name = "1D Burgers"
    config.data_config.nu = 0.01 / np.pi

    # Sample configuration
    sample_config = ml_collections.ConfigDict()
    sample_config.num_interior = {"t": 100, "x": 100}
    sample_config.num_boundary = {"t": 50, "x": 2}
    sample_config.num_initial = {"x": 100}
    sample_config.t_range = [0.0, 1.0]
    sample_config.x_range = [-1.0, 1.0]
    sample_config.rng_seed = 42
    config.data_config.sample_config = sample_config
    return config
