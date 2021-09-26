import numpy as np


class Point3D:

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    @classmethod
    def from_tuple(cls, tuple):
        return cls(tuple[0], tuple[1], tuple[2])

    @classmethod
    def from_np_array(cls, np_array):
        return cls(np_array[0], np_array[1], np_array[2])

    def to_np_array(self):
        return np.array([self.x, self.y, self.z])

    def to_list(self):
        return [self.x, self.y, self.z]

    def to_tuple(self):
        return (self.x, self.y, self.z)

    def to_int_tuple(self):
        return (int(round(self.x)), int(round(self.y)), int(round(self.z)))

    def distance_to(self, point):
        assert isinstance(point, Point3D), "Input must be a 3D Point!"
        distance = np.linalg.norm(self.to_np_array() - point.to_np_array())
        return distance


class Point2D:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    @classmethod
    def from_tuple(cls, tuple):
        return cls(tuple[0], tuple[1])

    @classmethod
    def from_np_array(cls, np_array):
        return cls(np_array[0], np_array[1])

    def to_np_array(self):
        return np.array([self.x, self.y])

    def to_list(self):
        return [self.x, self.y]

    def to_tuple(self):
        return (self.x, self.y)

    def to_int_tuple(self):
        return (int(round(self.x)), int(round(self.y)))

    def distance_to(self, point):
        assert isinstance(point, Point2D), "Input must be a 2D Point!"
        distance = np.linalg.norm(self.to_np_array() - point.to_np_array())
        return distance


class Pixel(Point2D):
    def __init__(self, u, v, value=None):
        super().__init__(u, v)
        self.u = u
        self.v = v
        self.value = value
