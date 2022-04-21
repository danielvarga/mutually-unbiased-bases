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
    for i in range(3):
        firstrow, phi = slicepair_data[i]
        sx0_sym = szollosi_modified_sym(firstrow, phi)
        sx1_sym = conjugate_pair(sx0_sym)
        c[2 * i, :, :] = sx0_sym
        c[2 * i + 1, :, :] = sx1_sym
    return ImmutableDenseNDimArray(c)


def apply_elemwise(fn, matrix):
    m = matrix.as_mutable()
    for k in range(m.shape[0]):
        for l in range(m.shape[1]):
            m[k, l] = fn(m[k, l])
    return m


def enforce_norm_one(p, variables):
    for var in variables:
        p = p.subs(conjugate(var) * var, 1)
    return p


def evaluate(s):
    return np.array(s).astype(np.complex128)


def substitute_slicepair_data(formula, slicepair_data_sym, slicepair_data):
    for (row_sym, phi_sym), (row, phi) in zip(slicepair_data_sym, slicepair_data):
        formula = formula.subs(phi_sym, phi)
        for elem_sym, elem in zip(row_sym, row):
            formula = formula.subs(elem_sym, elem)
    return formula


def collect_constraints(m):
    constraints = []
    rows, cols = m.shape
    for i in range(rows):
        for j in range(cols):
            if m[i, j] != 0:
                constraints.append(m[i, j])
    return constraints


def collect_slicepair_data(c):
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
    return slicepair_data



def create_symbolic_cube():
    # sy as in y-directional slice, a Fourier matrix.
    sy_sym = Matrix([[symbols(f'f_{i+1}{j+1}') for j in range(6)] for i in range(6)])
    phis = symbols('phi_1 phi_2 phi_3')
    variables = [symbols(f'f_{i+1}{j+1}') for j in range(6) for i in range(6)] + list(phis)


    slicepair_data_sym = [(sy_sym[2 * i, :], phis[i]) for i in range(3)]
    c_sym = build_cube_from_slicepair_data(slicepair_data_sym)
    return c_sym, sy_sym, phis, variables, slicepair_data_sym


def remove_redundant_constraints(constraints):
    reduced_constraints = []
    for cons1 in constraints:
        usable = True
        for cons2 in reduced_constraints:
            if cons2 - cons1 == 0:
                usable = False
                break
            if cons2 - conjugate(cons1) == 0:
                usable = False
                break
        if usable:
            reduced_constraints.append(cons1)
    return reduced_constraints


# c is actually only used to verify the constraints.
def extract_constraints_from_symbolic_cube(c):
    slicepair_data = collect_slicepair_data(c)
    c_semisym = build_cube_from_slicepair_data(slicepair_data)
    assert np.allclose(evaluate(c_semisym), c, atol=1e-4)

    c_sym, sy_sym, phis, variables, slicepair_data_sym = create_symbolic_cube()

    c_semisym_again = substitute_slicepair_data(c_sym, slicepair_data_sym, slicepair_data)
    assert np.allclose(evaluate(c_semisym_again), c, atol=1e-4)

    oned_constraints = []
    for i in range(6):
        for j in range(6):
            for oned in [c_sym[i, j, :], c_sym[:, i, j], c_sym[j, :, i]]:
                # assert np.isclose(evaluate(substitute_slicepair_data(sum(oned), slicepair_data_sym, slicepair_data)), 1, atol=1e-4)
                oned_constraints.append(sum(oned) - 1)

    unitarity_constraints = []
    for i in range(6):
        sx_reconstructed_sym = Matrix(c_sym[i, :, :])
        sy_reconstructed_sym = Matrix(c_sym[:, i, :])
        sz_reconstructed_sym = Matrix(c_sym[:, :, i])
        print("sx", sx_reconstructed_sym)
        print("----")
        print("sy", sy_reconstructed_sym)
        print("----")
        print("sz", sz_reconstructed_sym)

        assert np.allclose(evaluate(substitute_slicepair_data(sx_reconstructed_sym, slicepair_data_sym, slicepair_data)), c[i, :, :], atol=1e-4)
        assert np.allclose(evaluate(substitute_slicepair_data(sy_reconstructed_sym, slicepair_data_sym, slicepair_data)), c[:, i, :], atol=1e-4)
        assert np.allclose(evaluate(substitute_slicepair_data(sz_reconstructed_sym, slicepair_data_sym, slicepair_data)), c[:, :, i], atol=1e-4)


        prodx = Dagger(sx_reconstructed_sym) @ sx_reconstructed_sym - eye(6) * 6
        prodx = enforce_norm_one(prodx, variables)
        print("sx^* sx - 6Id", prodx)
        prodx_semisym = substitute_slicepair_data(prodx, slicepair_data_sym, slicepair_data)
        print(evaluate(prodx_semisym))


        prody = Dagger(sy_reconstructed_sym) @ sy_reconstructed_sym - eye(6) * 6
        prody = enforce_norm_one(prody, variables) / 2
        print("(sy^* sy - 6Id)/2", prody)
        prody_semisym = substitute_slicepair_data(prody, slicepair_data_sym, slicepair_data)
        print(evaluate(prody_semisym))


        prodz = Dagger(sz_reconstructed_sym) @ sz_reconstructed_sym - eye(6) * 6
        prodz = enforce_norm_one(prodz, variables) / 2
        print("(sz^* sz - 6Id)/2", prodz)
        prodz_semisym = substitute_slicepair_data(prodz, slicepair_data_sym, slicepair_data)
        print(evaluate(prodz_semisym))

        x_unitarity_constraints = collect_constraints(prodx)
        y_unitarity_constraints = collect_constraints(prody)
        z_unitarity_constraints = collect_constraints(prodz)

        unitarity_constraints += x_unitarity_constraints + y_unitarity_constraints + z_unitarity_constraints

    constraints = unitarity_constraints + oned_constraints
    constraints = remove_redundant_constraints(constraints)
    return constraints



constraints = extract_constraints_from_symbolic_cube(c)

for cons in constraints:
    print(latex(Eq(cons, 0)))
