import ml_collections
import numpy as np
from torch.utils.data import IterableDataset

from .geometry import Rectangle1D


class DataGenerator1D(IterableDataset):
    def __init__(
        self,
        data_cfg: ml_collections.ConfigDict,
    ):
        self.domain_x = Rectangle1D(min=data_cfg.x_range[0], max=data_cfg.x_range[1])
        self.domain_t = Rectangle1D(min=data_cfg.t_range[0], max=data_cfg.t_range[1])
        self.rng = np.random.default_rng(data_cfg.rng_seed)
        self.cfg = data_cfg

    def __iter__(self):
        return self

    def __next__(self):
        np_dict = {
            "interior": {
                "t": self.domain_t.sample_interior(self.cfg.num_interior, self.rng),
                "x": self.domain_x.sample_interior(self.cfg.num_interior, self.rng),
            },
            "boundary": {
                "t": self.domain_t.sample_interior(self.cfg.num_boundary, self.rng),
                "x": self.domain_x.sample_boundary(self.cfg.num_boundary, self.rng),
            },
            "initial": {
                "t": np.ones((self.cfg.num_initial,), dtype=np.float32)
                * self.cfg.t_range[0],
                "x": self.domain_x.sample_interior(self.cfg.num_initial, self.rng),
            },
        }

        return np_dict
