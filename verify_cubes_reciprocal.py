import sys
import numpy as np
from base import *


# c = np.load("canonized_cubes/canonized_cube_00000.npy")

# data/cube_d6.00001.reciprocal.npy
c = np.load(sys.argv[1])


# def verify_sum(v, atol=1e-4):
#     print(v.sum())


def verify_cube_properties(c, atol=1e-4):
    n = c.shape[0]
    for i in range(n):
        # print("verifying 2D slices", i)
        verify_hadamard(c[i, :, :], atol=atol)
        verify_hadamard(c[:, i, :], atol=atol)
        verify_hadamard(c[:, :, i], atol=atol)

    for i in range(n):
        # print("verifying equivalence of parallel slices", i)
        b1 = c[0, :, :]
        b2 = c[i, :, :]
        verify_phase_equivalence(b1, b2, atol=atol)
        b1 = c[:, 0, :]
        b2 = c[:, i, :]
        verify_phase_equivalence(b1, b2, atol=atol)
        b1 = c[:, :, 0]
        b2 = c[:, :, i]
        verify_phase_equivalence(b1, b2, atol=atol)

    try:
        for i in range(n):
            for j in range(n):
                # print("verifying 1D slices", i, j)
                verify_sum(c[i, j, :], atol=atol)
                verify_sum(c[:, i, j], atol=atol)
                verify_sum(c[j, :, i], atol=atol)
    except:
        print("1d sum failed at index", i, j)
        exit()




verify_cube_properties(c, atol=1e-3)
