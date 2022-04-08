import sys
import numpy as np
from itertools import permutations
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D

from base import *


def verify_sum(v):
    print(np.abs(v.sum()))
    # assert np.isclose(v.sum(), 1, atol=2e-4)

# np.set_printoptions(precision=12, suppress=True, linewidth=100000)
np.set_printoptions(precision=5, suppress=True)


def verify_cube_properties(c):
    for i in range(6):
        # print("verifying 2D slices", i)
        verify_hadamard(c[i, :, :])
        verify_hadamard(c[:, i, :])
        verify_hadamard(c[:, :, i])

    for i in range(6):
        for j in range(6):
            # print("verifying 1D slices", i, j)
            verify_sum(c[i, j, :])
            verify_sum(c[:, i, j])
            verify_sum(c[j, :, i])


filename = sys.argv[1]

from_triplet = True
if from_triplet:
    # triplets/triplet_mub_00018.npy
    a = np.load(filename)
    c = hadamard_cube(a)
else:
    # cube_00001.npy
    filename = sys.argv[1]
    c = np.load(filename)


verify_cube_properties(c)
