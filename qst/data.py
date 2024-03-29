import itertools as itr
import numpy as np
import time
from scipy import linalg
from .operators import dagger, Rotation


## This particular implementation is taken directly from
## https://github.com/emerali/rand_wvfn_sampler.git

def kron_mv_low_mem(x, *matrices):
    n = [m.shape[0] for m in matrices]
    l = np.prod(n)
    r = 1
    V = x.astype(complex)
    for s in range(len(n))[::-1]:
        l //= n[s]
        m = matrices[s]
        for k in range(l):
            for i in range(r):
                slc = slice(k*n[s]*r + i, (k+1)*n[s]*r + i, r)
                U = V[slc]
                V[slc] = np.dot(m, U)
        r *= n[s]

    return V


dflt_unitary_dict = {
    "X": Rotation([0, 1, 0], -np.pi / 4),
    "Y": Rotation([1, 0, 0], np.pi / 4),
    "Z": Rotation([0, 0, 1], 0),
    "B": np.array([[1, 0, 0, 1], [0, 1, 1, 0], [0, 1, -1, 0], [1, 0, 0, -1]])
    / np.sqrt(2),
}

dflt_unitary_dict_matrix = {
    "X": Rotation([0, 1, 0], -np.pi / 4).R,
    "Y": Rotation([1, 0, 0], np.pi / 4).R,
    "Z": Rotation([0, 0, 1], 0).R,
    "B": np.array([[1, 0, 0, 1], [0, 1, 1, 0], [0, 1, -1, 0], [1, 0, 0, -1]])
    / np.sqrt(2),
}


def dflt_unique_bases(N):
    XYZ = np.array(["X", "Y", "Z"])
    unique_bases = XYZ[
        np.fromiter(itr.chain(*itr.product([0, 1, 2], repeat=N)), dtype=np.int)
    ].reshape(-1, N)
    return unique_bases


def basistounitary(basis, unitary_dict=dflt_unitary_dict, rotation_error=[0.01, 0.01]):
    rotation_error = np.array(rotation_error)

    unitary_keys = list(unitary_dict.keys())
    unitary_vals = np.array(list(unitary_dict.values()))

    idcs = [unitary_keys.index(i) for i in basis]
    Us = unitary_vals[idcs]

    Us = np.array(
        [
            Us[i].perturb(rotation_error[0], rotation_error[1])
            if type(Us[i]) == Rotation
            else Us[i]
            for i in range(len(Us))
        ]
    )
    Us_ = np.copy(Us)
    if type(Us[0]) == Rotation:
        Us_[0] = Us[0].R
    else:
        Us_[0] = Us[0]
    for i in range(1, len(Us[1:])+1):
        if type(Us[i]) == Rotation:
            Us_[i] = Us[i].R
        else:
            Us_[i] = Us[i]
    return Us_


def sample(N, psi, basis=None, unitary_dict=dflt_unitary_dict, rotation_error=[0.01, 0.01]):
    if basis is None:
        basis = "Z" * N
    Us = basistounitary(basis, unitary_dict, rotation_error)
    p = np.square(np.absolute(kron_mv_low_mem(psi, *Us)))
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
    data = np.array(np.logical_xor(data, flip), dtype=int)
    return data


def create_data(
    D, N, psi, unique_bases=None, unitary_dict=dflt_unitary_dict, rotation_error=[0.01, 0.01]
):
    if unique_bases is None:
        unique_bases = dflt_unique_bases(N)

    bases = []
    data = []
    for i in unique_bases:
        for j in range(D):
            bases.append(" ".join(i))
            data.append(sample(N, psi, i, unitary_dict, rotation_error))
    bases = np.array(bases)
    data = np.array(data)
    data = data.reshape(-1, N)
    rho = np.outer(psi, np.conj(psi))
    return data, rho, bases, unique_bases
