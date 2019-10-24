import os
import sys

path = os.path.dirname(__file__)
sys.path.append(path)

import itertools as itr
import numpy as np
import qst

N = 2

i = np.complex(0, 1)

rho = np.kron(np.array([[1, -i], [i, 1]]) / 2, np.array([[1, 0], [0, 0]]))

unique_bases = np.array([["X", "X"], ["Z", "X"], ["X", "Z"], ["Y", "Y"], ["Z", "Z"]])

data, rho, bases, unique_bases = qst.data.create_data(
    1000,
    N,
    rho,
    unique_bases,
    unitary_dict=qst.data.dflt_unitary_dict,
    rotation_error=[0.01, 0.01],
)

data = qst.data.measurement_error(data, 0.01, 0.02)

converted_rho = np.array([np.real(rho), np.imag(rho)])

unitary_dict = qst.data.dflt_unitary_dict_matrix

data_path = path + "/data/data 1.txt"
np.savetxt(data_path, data)

rho_path = path + "/data/rho 1"
np.save(rho_path, converted_rho)

bases_path = path + "/data/bases 1.txt"
np.savetxt(bases_path, bases, delimiter=" ", fmt="%s")

unique_bases_path = path + "/data/unique_bases 1.txt"
np.savetxt(unique_bases_path, unique_bases, delimiter=" ", fmt="%s")
