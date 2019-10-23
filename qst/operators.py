import numpy as np
from scipy import linalg


def dagger(O):
    return np.transpose(np.conjugate(O))


class Rotation(object):
    def __init__(self, axis, angle):
        super(Rotation, self).__init__()

        axis = np.array(axis)

        self.axis = axis / np.sqrt(np.sum(axis ** 2))
        self.angle = angle

        i = np.complex(0, 1)
        sigma = np.array([[[0, 1], [1, 0]], [[0, -i], [i, 0]], [[1, 0], [0, -1]]])
        R = linalg.expm(-i * np.einsum("imn,i->mn", sigma, axis) * angle)

        self.R = R
        pass

    def __repr__(self):
        return self.R.__repr__()

    def __str__(self):
        return self.R.__str__()

    def dagger(self):
        return Rotation(self.axis, -self.angle)

    def perturb(self, axis_error, angle_error):
        axis = self.axis
        angle = self.angle

        random_axis = np.random.rand(3)
        orthogonal_axis = random_axis - np.dot(random_axis, axis) * axis
        orthonormal_axis = orthogonal_axis / np.sqrt(np.sum(orthogonal_axis ** 2))

        epsilon_1 = np.abs(np.random.normal(0, axis_error))
        perturbed_axis = (1 - epsilon_1) * axis + epsilon_1 * orthonormal_axis

        epsilon_2 = np.random.normal(0, angle_error)
        perturbed_angle = (1 + epsilon_2) * angle

        return Rotation(perturbed_axis, perturbed_angle)

    pass
