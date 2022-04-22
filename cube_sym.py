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


def truncate_canon(g):
    F = canonical_fourier(g['x'], g['y'])
    gi = invert(g)
    # The magnitude of gi['d_r'] is 1/sqrt(6), we have to put that
    # back if we drop gi['d_r']
    return gi['d_l'] @ gi['p_l'] @ F / 6 ** 0.5


b1_canon = get_canonizer(a[1])
a[1] = truncate_canon(b1_canon)
b2_canon = get_canonizer(a[2])
a[2] = truncate_canon(b2_canon)

verify_mub(a)
c = hadamard_cube(a)
verify_cube_properties(c)


b1_canon = get_canonizer(a[1])
b2_canon = get_canonizer(a[2])
print(b1_canon)
print(b2_canon)


def is_row_structure_compatible(b1, b2):
    result_1 = find_blocks(b1, allow_transposal=False)
    bipart_col_1, tripart_col_1, is_transposed_1 = result_1
    result_2 = find_blocks(b2, allow_transposal=False)
    bipart_col_2, tripart_col_2, is_transposed_2 = result_2
    return tripart_col_1 == tripart_col_2


def reorder_mub(a_orig):
    a = a_orig.copy()
    assert is_row_structure_compatible(a[1], a[2])
    # if yes, we can safely apply it individually to permute rows.
    a[1] = arrange_blocks(a[1])
    a[2] = arrange_blocks(a[2])
    verify_mub(a)
    c = hadamard_cube(a)
    verify_cube_properties(c)
    return a


# apply at most 2 transpositons to a[2] columns
# so that sx becomes block-circulant.
# permuting a[2] columns permutes sx columns,
# permuting a[1] columns permutes sx rows.
def turn_blocks_to_circulant(b):
    raise "unimplemented"


a = reorder_mub(a)
c = hadamard_cube(a)
print(angler(c[0, :3, :3]))
print(angler(c))
print("----")
a[1] = a[1][:, [0, 2, 1, 3, 4, 5]]
verify_mub(a)
c = hadamard_cube(a)
verify_cube_properties(c)
print(angler(c[0, :3, :3]))
print(angler(c))

exit()


sx = c[0, :, :]
sy = c[:, 0, :]
sz = c[:, :, 0]


a[1] = arrange_blocks(a[1])
a[2] = arrange_blocks(a[2])

verify_mub(a)

c = hadamard_cube(a)

verify_cube_properties(c)


print(angler(c))

exit()


# g = {'d_l' : Dl, 'd_r' : Dr, 'p_l' : row_perm, 'p_r' : col_perm, 'x' : p[0], 'y' : p[1]}

prod = trans(a[1], a[2])

print(is_phase_equivalent(sx, prod))
print(is_phase_equivalent(sy, a[2]))
print(is_phase_equivalent(sz, a[1]))

print(is_equivalent(sx, prod))
print(is_equivalent(sy, a[2]))
print(is_equivalent(sz, a[1]))

print("sx dephased", complex_dephase(sx))
print("prod dephased", complex_dephase(prod))
print("div", complex_dephase(prod)[0] / complex_dephase(sx)[0])

print("sy dephased", complex_dephase(sy))
print("a[2] dephased", complex_dephase(np.conjugate(a[2])))
print("div", complex_dephase(np.conjugate(a[2]))[0] / complex_dephase(sy)[0])

print("sz dephased", complex_dephase(sz))
print("a[1] dephased", complex_dephase(a[1]))
print("div", complex_dephase(a[1])[0] / complex_dephase(sz)[0])

print("sy0", get_canonizer(sy))
print("sz0", get_canonizer(sz))
print("a", get_canonizer(a[1]))
print("b", get_canonizer(a[2]))
exit()







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
    for direction in range(3):
        for coord in range(6):

            if direction == 0:
                # we observe that for the sx slices,
                # the odd ones give the same constraints as the even ones.
                # to save running time, we don't even extract the odd ones:
                if coord % 2 == 1:
                    continue
            else:
                # we observe that for the sy and sz slices,
                # one slice is enough, the rest give redundant constraints.
                # moreover the sy and the sz slice gives the same constraints, but
                # we do not hardwire this here:
                if coord > 0:
                    continue

            s_sym = Matrix(slic(c_sym, direction, coord))
            s = slic(c, direction, coord)
            assert np.allclose(evaluate(substitute_slicepair_data(s_sym, slicepair_data_sym, slicepair_data)), s, atol=1e-4)

            prod = Dagger(s_sym) @ s_sym - eye(6) * 6
            prod = enforce_norm_one(prod, variables)

            # we observe that sy and sz unitarity constraints are always divisible by 2.
            if direction !=0:
                prod /= 2

            prod_semisym = substitute_slicepair_data(prod, slicepair_data_sym, slicepair_data)
            assert np.allclose(evaluate(prod_semisym), 0, atol=1e-4)

            unitarity_constraints += collect_constraints(prod)

    constraints = unitarity_constraints + oned_constraints
    constraints = remove_redundant_constraints(constraints)
    return constraints



constraints = extract_constraints_from_symbolic_cube(c)

print("Collected constraints:")
for cons in constraints:
    print(latex(Eq(cons, 0)))
