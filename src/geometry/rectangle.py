import numpy as np

from .base import GeometryBase


class Rectangle1D(GeometryBase):
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def sample_boundary(self, num_samples=1, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        coords = rng.choice([self.min, self.max], size=(num_samples,))
        return np.array(coords, dtype=np.float32)

    def sample_interior(self, num_samples, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        coords = rng.uniform(low=self.min, high=self.max, size=(num_samples,))
        return np.array(coords, dtype=np.float32)


class Rectangle2D(GeometryBase):
    def __init__(self, xmin, xmax, ymin, ymax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    @property
    def width(self):
        return self.xmax - self.xmin

    @property
    def height(self):
        return self.ymax - self.ymin

    def sample_boundary(self, num_samples, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        array = rng.uniform(low=0.0, high=1, size=(num_samples,))

        num_list = [
            int(self.width / (2 * (self.width + self.height)) * num_samples),
            int(self.width / (2 * (self.width + self.height)) * num_samples),
            int(self.height / (2 * (self.width + self.height)) * num_samples),
        ]
        num_list.append(num_samples - sum(num_list))
        num_list = np.cumsum(num_list)

        coords = np.zeros((num_samples, 2))

        coords[0 : num_list[0], 0] = self.xmin
        coords[0 : num_list[0], 1] = self.ymin + array[0 : num_list[0]] * self.height
        coords[num_list[0] : num_list[1], 0] = (
            self.xmin + array[num_list[0] : num_list[1]] * self.width
        )
        coords[num_list[0] : num_list[1], 1] = self.ymax
        coords[num_list[1] : num_list[2], 0] = self.xmax
        coords[num_list[1] : num_list[2], 1] = (
            self.ymax - array[num_list[1] : num_list[2]] * self.height
        )
        coords[num_list[2] : num_list[3], 0] = (
            self.xmax - array[num_list[2] : num_list[3]] * self.width
        )
        coords[num_list[2] : num_list[3], 1] = self.ymin

        return coords

    def sample_interior(self, num_samples, rng=None, tol=1e-9):
        if rng is None:
            rng = np.random.default_rng()
        coords = rng.uniform(
            low=[self.xmin + tol, self.ymin + tol],
            high=[self.xmax, self.ymax],
            size=(num_samples, 2),
        )

        return coords
