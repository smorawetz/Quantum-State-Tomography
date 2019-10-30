import os
import sys

path = os.path.dirname(__file__)
sys.path.append(path)

import itertools as itr
import numpy as np
import time
import qst

# N_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
N_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

i = np.complex(0, 1)

states_list = [ np.array([1, i]) / np.sqrt(2), np.array([1, 0]), np.array([0, 1]), np.array([1, 1]) / np.sqrt(2), np.array([1, -1]) / np.sqrt(2),
    np.array([1, 0]), np.array([1, -i]) / np.sqrt(2), np.array([1, 1]) / np.sqrt(2), np.array([0, 1]), np.array([1, i]) / np.sqrt(2) ]

states_list_conj = list(map(np.conj, states_list))


def gen_data(N, states_list):

    time1 = time.time()

    psi_list = states_list[:N]

    for index in range(N - 1):
        psi_list[index+1] = np.tensordot(psi_list[index], psi_list[index+1], axes=0).flatten() 

    psi = psi_list[-1]
    
    unique_bases = []
    unique_bases.append("Z" * N)
    for basis_num in range(N):
        X_basis = "Z" * basis_num + "X" + "Z" * (N - basis_num - 1)
        Y_basis = "Z" * basis_num + "Y" + "Z" * (N - basis_num - 1)
        unique_bases.append(X_basis)
        unique_bases.append(Y_basis)

    data, rho, bases, unique_bases = qst.data.create_data(
        1000,
        N,
        psi,
        unique_bases,
        unitary_dict=qst.data.dflt_unitary_dict,
        rotation_error=[0.01, 0.01],
    )

    data = qst.data.measurement_error(data, 0.01, 0.02)
    
    converted_rho = np.array([np.real(rho), np.imag(rho)])
    
    unitary_dict = qst.data.dflt_unitary_dict_matrix
    
    data_path = path + "./data/N{0}_1000_data.txt".format(N)
    np.savetxt(data_path, data)
    
    rho_path = path + "./data/N{0}_rho".format(N)
    np.save(rho_path, converted_rho)
    
    bases_path = path + "./data/N{0}_1000_bases.txt".format(N)
    np.savetxt(bases_path, bases, delimiter=" ", fmt="%s")
    
    unique_bases_path = path + "./data/N{0}_unique_bases.txt".format(N)
    np.savetxt(unique_bases_path, unique_bases, delimiter=" ", fmt="%s")

    time2 = time.time()

    print("N = ", N, " took ", time2 - time1, " seconds")

for N in N_list:
    gen_data(N, states_list)
