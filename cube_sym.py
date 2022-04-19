import sys
import numpy as np
from itertools import permutations
import matplotlib.pyplot as plt

from base import *


np.set_printoptions(precision=5, suppress=True, linewidth=160)

# triplet_mub_00018.npy
filename = sys.argv[1]
a = np.load(filename)

a = np.stack([np.eye(6, dtype=np.complex128), a[0], a[1]])

for i in range(1, len(a)):
    verify_hadamard(a[i])

verify_mub(a)

c = hadamard_cube(a)

verify_cube_properties(c)

if filename.endswith("triplet_mub_00018.npy"):
    x_perm = [0, 1, 2, 3, 4, 5]
    y_perm = [0, 4, 1, 2, 3, 5]
    z_perm = [0, 4, 2, 3, 5, 1]
elif filename.endswith("triplet_mub_00171.npy"):
    x_perm = [0, 2, 1, 5, 3, 4]
    y_perm = [0, 1, 2, 3, 4, 5]
    z_perm = [0, 5, 1, 2, 4, 3]
else:
    print("WARNING: no manually computed permutations available")
    x_perm = y_perm = z_perm = range(6)
c = c[x_perm, :, :]
c = c[:, y_perm, :]
c = c[:, :, z_perm]


# that's the extra cyclic permutation of the lower half of the Szollosi
# that makes it diag([1 1 1 phi phi phi]) times the textbook version.
if filename == "triplets/triplet_mub_00018.npy":
    c = c[:, [0, 1, 2, 5, 3, 4], :]
elif filename.endswith("triplet_mub_00171.npy"):
    pass # good already


def recreate_slice(firstrow, phi):
    blockcirculant = szollosi_original(firstrow)
    blockcirculant[3:, :] *= phi
    return blockcirculant


def deconstruct_slice(sx, atol=1e-4):
    firstrow = sx[0, :]
    blockcirculant = szollosi_original(firstrow)
    ratio = sx / blockcirculant
    assert np.allclose(ratio[:3, :], 1, atol=atol)
    phi = np.mean(ratio[3:, :])
    ratio[3:, :] /= phi
    assert np.allclose(ratio[3:, :], 1, atol=atol)
    assert np.allclose(recreate_slice(firstrow, phi), sx, atol=atol)
    return firstrow, phi


for indx in range(6):
    sx = c[indx, :, :]
    firstrow, phi = deconstruct_slice(sx)
    # print(angler(recreate_slice(firstrow, phi)))
    print(angler(phi))
