import itertools as itr
import numpy as np
from scipy import linalg
from .operators import dagger, Rotation

dflt_unitary_dict = {
    "X": Rotation([0, 1, 0], -np.pi / 4),
    "Y": Rotation([1, 0, 0], np.pi / 4),
    "Z": Rotation([0, 0, 1], 0),
}

dflt_unitary_dict_matrix = {
    "X": Rotation([0, 1, 0], -np.pi / 4).R,
    "Y": Rotation([1, 0, 0], np.pi / 4).R,
    "Z": Rotation([0, 0, 1], 0).R,
}


def dflt_unique_bases(N):
    XYZ = np.array(["X", "Y", "Z"])
    unique_bases = XYZ[
        np.fromiter(itr.chain(*itr.product([0, 1, 2], repeat=N)), dtype=np.int)
    ].reshape(-1, N)
    return unique_bases


def basistounitary(basis, unitary_dict=dflt_unitary_dict, rotation_error=[0, 0]):
    rotation_error = np.array(rotation_error)

    unitary_keys = list(unitary_dict.keys())
    unitary_vals = np.array(list(unitary_dict.values()))

    idcs = [unitary_keys.index(i) for i in basis]
    Us = unitary_vals[idcs]

    Us = np.array(
        [Us[i].perturb(rotation_error[0], rotation_error[1]) for i in range(len(Us))]
    )

    U = Us[0].R
    for i in Us[1:]:
        U = np.kron(U, i.R)
    return U


def sample(N, rho, basis=None, unitary_dict=dflt_unitary_dict, rotation_error=[0, 0]):
    if basis is None:
        basis = "Z" * N
    U = basistounitary(basis, unitary_dict, rotation_error)
    p = np.real(np.diagonal(np.einsum("ij,jk,kl->il", U, rho, dagger(U))))
    cum_p = np.array([np.sum(p[: i + 1]) for i in range(2 ** N)])
    sample = np.random.rand()
    sample = np.sum(cum_p < sample)
    sample = np.array(list(np.binary_repr(sample, width=N)), dtype=int)
    return sample


def measurement_error(data, p0, p1):
    r = np.random.rand(*data.shape)
    flip = np.logical_or(
        np.logical_and(data == 0, r < p0), np.logical_and(data == 1, r < p1)
    )
    data = np.array(np.logical_xor(data, flip),dtype=int)
    return data


def create_data(
    D, N, rho, unique_bases=None, unitary_dict=dflt_unitary_dict, rotation_error=[0, 0]
):
    if unique_bases is None:
        unique_bases = dflt_unique_bases(N)

    bases = []
    data = []
    for i in unique_bases:
        for j in range(D):
            bases.append(i)
            data.append(sample(N, rho, i, unitary_dict, rotation_error))
    bases = np.array(bases)
    data = np.array(data)
    data = data.reshape(-1, N)
    return data, rho, bases, unique_bases
