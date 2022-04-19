import sys
import numpy as np
from itertools import permutations
import matplotlib.pyplot as plt

from sympy import *
from sympy.physics.quantum.dagger import Dagger
from sympy.tensor.array import MutableDenseNDimArray

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


# beware: elements of firstrow have abs 1/sqrt(6),
# but phi has abs 1.
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


def circulant_sym(firstrow):
    a, b, c = firstrow
    return Matrix([[a, b, c], [c, a, b], [b, c, a]])


def szollosi_original_sym(firstrow):
    a, b, c, d, e, f = firstrow
    block1 = circulant_sym([a, b, c])
    block2 = circulant_sym([d, e, f])
    block3 = circulant_sym([conjugate(d), conjugate(f), conjugate(e)])
    block4 = circulant_sym([-conjugate(a), -conjugate(c), -conjugate(b)])
    blockcirculant = Matrix(BlockMatrix([[block1, block2], [block3, block4]]))
    return blockcirculant


def szollosi_modified_sym(firstrow, phi):
    blockcirculant = szollosi_original_sym(firstrow)
    blockcirculant[3:, :] *= phi
    return blockcirculant


def conjugate_pair_sym(sx):
    sxb = sx.copy()
    b00 = Dagger(sx[:3, :3])
    b01 = Dagger(sx[:3, 3:])
    b10 = Dagger(sx[3:, :3])
    b11 = Dagger(sx[3:, 3:])
    sxb[:3, :3] = b11
    sxb[:3, 3:] = b10
    sxb[3:, :3] = b01
    sxb[:3, :3] = b00
    return sxb


def build_cube_from_slicepair_data(slicepair_data):
    c = MutableDenseNDimArray([0] * 216, (6, 6, 6))
    print(c.shape)
    for i in range(3):
        firstrow, phi = slicepair_data[i]
        sx0_sym = szollosi_modified_sym(firstrow, phi)
        sx1_sym = conjugate_pair(sx0_sym)
        c[2 * i, :, :] = sx0_sym
        c[2 * i + 1, :, :] = sx1_sym
    return c


def evaluate(s):
    return np.array(s).astype(np.complex128)


slicepair_data = []
for indx in range(0, 6, 2):
    sx0 = c[indx, :, :]
    sx1 = c[indx + 1, :, :]
    assert np.allclose(sx1, conjugate_pair(sx0), atol=1e-4)

    firstrow, phi = deconstruct_slice(sx0)
    slicepair_data.append((firstrow, phi))
    sx0_sym = szollosi_modified_sym(firstrow, phi)
    sx0_recalc = evaluate(sx0_sym)
    assert np.allclose(sx0_recalc, sx0, atol=1e-4)

    sx1_sym = conjugate_pair(sx0_sym)
    sx1_recalc = evaluate(sx1_sym)
    assert np.allclose(sx1_recalc, sx1, atol=1e-4)


c_sym = build_cube_from_slicepair_data(slicepair_data)
assert np.allclose(evaluate(c_sym), c, atol=1e-4)
