import ml_collections
import numpy as np
from torch.utils.data import IterableDataset

from .geometry import Rectangle1D, cartesian_product


class DataGenerator1D(IterableDataset):
    def __init__(
        self,
        data_cfg: ml_collections.ConfigDict,
    ):
        self.domain_x = Rectangle1D(min=data_cfg.x_range[0], max=data_cfg.x_range[1])
        self.domain_t = Rectangle1D(min=data_cfg.t_range[0], max=data_cfg.t_range[1])
        self.rng = np.random.default_rng(data_cfg.rng_seed)
        self.cfg = data_cfg.sample_config

    def __iter__(self):
        return self

    def __next__(self):
        np_dict = {
            "interior": cartesian_product(
                self.domain_t.sample_interior(self.cfg.num_interior["t"], self.rng),
                self.domain_x.sample_interior(self.cfg.num_interior["x"], self.rng),
            ),
            "boundary": cartesian_product(
                self.domain_t.sample_interior(self.cfg.num_boundary["t"], self.rng),
                self.domain_x.sample_boundary(self.cfg.num_boundary["x"], self.rng),
            ),
            "initial": cartesian_product(
                np.array(self.cfg.t_range),
                self.domain_x.sample_interior(self.cfg.num_initial["x"], self.rng),
            ),
        }
        np_dict = {k: np.reshape(v, (-1, 2)) for k, v in np_dict.items()}

        return np_dict
