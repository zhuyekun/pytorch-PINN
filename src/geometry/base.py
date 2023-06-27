import abc

class GeometryBase(abc.ABC):

    @abc.abstractmethod
    def sample_boundary(self):
        pass

    @abc.abstractmethod
    def sample_interior(self):
        pass