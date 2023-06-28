import ml_collections


def config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    config.model_config = ml_collections.ConfigDict()
    config.train_config = ml_collections.ConfigDict()
    config.data_config = ml_collections.ConfigDict()

    config.train_config.batch_size = 32
    config.train_config.epochs = 100
    config.train_config.lr = 1e-3
    config.train_config.weight_decay = 1e-5

    config.model_config.units = [2, 128, 128, 2]

    config.data_config.num_interior = 100
    config.data_config.t_range = [0.0, 1.0]
    config.data_config.x_range = [-1.0, 1.0]
    config.data_config.num_t = 50
    return config
