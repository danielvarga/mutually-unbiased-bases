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
    pass # verify_hadamard(a[i])

verify_mub(a)
c = hadamard_cube(a)
verify_cube_properties(c)


a_prime = cube_to_mub(c)
# a_prime = cube_to_mub_simplified(c)
verify_mub(a_prime)


# 00001
# print(angler(c[:, :, 0][:, [0, 2, 5, 1, 4, 3]][[2, 3, 4, 0, 5, 1], :]))
# 00075
# print(angler(c[0, :, :][:, [0, 3, 5, 1, 4, 2]][[5, 1, 2, 3, 4, 1], :]))
# print(angler(c[:, :, :][:, :, [0, 3, 5, 1, 4, 2]][:, [5, 1, 2, 3, 4, 1], :]))

c = c[:, :, :][:, :, [0, 3, 5, 1, 4, 2]][:, [5, 1, 2, 3, 4, 1], :]

sx = c[0, :, :]



# a[indx]^dag is left-applied to everybody. so that it turns into Id.
# then MUB elements are rotated among themselves
# so that a[index] becomes a[0]
def swap_identity(a, indx):
    if indx == 0:
        return a.copy()

    assert np.allclose(a[0], np.eye(6))
    a2 = a.copy()
    a2[1] = trans(a[indx], a[0])
    a2[2] = trans(a[indx], a[3 - indx])
    verify_mub(a2)
    return a2


def standardize_triplet_order(a):
    a_orig = a.copy()
    for i in range(3):
        a = swap_identity(a_orig, i)
        result1 = find_blocks(a[1], allow_transposal=False)
        result2 = find_blocks(a[2], allow_transposal=False)
        if result1 is not None and result2 is not None:
            # not transposed
            assert not result1[2] and not result2[2]
            return a
    return None


# a canonical hadamard has its bipartition of columns interlaced like 010101
# and its tripartition of rows interlaced like 012012.
# here we rearrange it into 000111, 001122.
def deinterlace(b_orig):
    b = b_orig[[0, 3, 1, 4, 2, 5], :]
    b = b[:, [0, 2, 4, 1, 3, 5]]
    return b


def hardwired_reorder(a_orig):
    a = a_orig.copy()
    a[1] = deinterlace(a[1])
    a[2] = deinterlace(a[2])
    return a


def mub_type(a):
    # by convention
    # counts[0] is number of Fourier,
    # counts[1] is number of Fourier-transposed
    # counts[2] is number of neither.
    counts = [0, 0, 0]
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            result = find_blocks(trans(a[i], a[j]), allow_transposal=True)
            if result is None:
                counts[2] += 1
            else:
                _, _, is_transposed = result
                counts[int(is_transposed)] += 1
    if counts == [6, 0, 0]:
        return "sporadic"
    elif counts == [2, 2, 2]:
        return "Szollosi"
    else:
        return "WTF"


def truncate_canon(g, also_left_perm=False):
    Id = np.eye(6, dtype=np.complex128)
    # The magnitude of gi['d_r'] is 1/sqrt(6), we have to put that
    # back if we drop gi['d_r']
    if also_left_perm:
        phases, perm_m = np.diag(g['d_l']), g['p_l']
        switched_perm_m, switched_phases = switch_phase_perm(phases, perm_m)
        return create_generalized_fourier(np.diag(switched_phases), Id, g['x'], g['y'], Id, Id / 6 ** 0.5)
    else:
        create_generalized_fourier(g['d_l'], g['p_l'], g['x'], g['y'], Id, Id / 6 ** 0.5)


def is_row_structure_compatible(b1, b2):
    result_1 = find_blocks(b1, allow_transposal=False)
    bipart_col_1, tripart_col_1, is_transposed_1 = result_1
    result_2 = find_blocks(b2, allow_transposal=False)
    bipart_col_2, tripart_col_2, is_transposed_2 = result_2
    return tripart_col_1 == tripart_col_2


# more percisely, find_permutation_that_turns_blocks_to_circulant()
# apply at most 2 transpositons to a[2] columns
# so that sx becomes block-circulant.
# permuting a[2] columns permutes sx columns,
# permuting a[1] columns permutes sx rows.
# assumes that the bipartition is standard (012)(345).
def turn_blocks_to_circulant(b):
    perm = list(range(6))
    w = visualize_clusters(b, group_conjugates=False)
    if w[2, 1] == w[1, 2]:
        perm[2], perm[1] = perm[1], perm[2]
    if w[4, 5] == w[5, 4]:
        perm[4], perm[5] = perm[5], perm[4]
    return perm


def check_phi_property(b, atol=1e-4):
    b2 = szollosi_original(b[0, :])
    ratio = b / b2
    phi_candidate = ratio[3, 0]
    ratio[3:, :] /= phi_candidate
    if np.allclose(ratio, 1, atol=atol):
        return phi_candidate
    else:
        return None


def turn_lower_half_to_single_phi(a_orig):
    a = a_orig.copy()
    for i in range(3):
        c = hadamard_cube(a)
        phi = check_phi_property(c[0, :, :])
        if phi is not None:
            return a, phi
        a[1] = a[1][:, [0, 1, 2, 4, 5, 3]]
        # print("rotating second triangle of a[1] to get constant phi")
    assert False, "none of the rotations leads to a constant phi"


# 1. rearranges the MUB columns and MUB rows so that
#    the cube has a block structure consisting of 3x3x2 blocks.
# 2. permutes within block so that each 3x3 is circulant.
# 3. permutes the lower half of the sx slices
#    (more exactly, the a[1] columns) so that the lower half
#    is constant phi times the Szollosi formula (6).
def reorder_mub(a_orig):
    a = a_orig.copy()
    assert is_row_structure_compatible(a[1], a[2])
    # if yes, we can safely apply it individually to permute rows.
    a[1] = arrange_blocks(a[1])
    a[2] = arrange_blocks(a[2])
    verify_mub(a)
    c = hadamard_cube(a)
    verify_cube_properties(c)

    perm = turn_blocks_to_circulant(c[0, :, :])
    a[2] = a[2][:, perm]

    c = hadamard_cube(a)
    verify_cube_properties(c)

    perm = turn_blocks_to_circulant(c[0, :, :])
    assert perm == list(range(6))

    a, phi = turn_lower_half_to_single_phi(a)

    return a


# does two things:
# 1. deduces their generalized F-form from the MUB elements,
#    and removes the superfluous right actions.
# 2. permutes the MUB rows and columns so that
#    the appropriate block-circulant structure and conjugate
#    pairing appears.
def standardize_mub(a):
    b1_canon = get_canonizer(a[1])
    b2_canon = get_canonizer(a[2])
    # assert np.all(b1_canon['p_l'] == b2_canon['p_l'])

    b1_truncated_canon = truncate_canon(b1_canon, also_left_perm=True)
    b2_truncated_canon = truncate_canon(b2_canon, also_left_perm=True)
    dl1 = b1_truncated_canon['d_l']
    b2_truncated_canon['d_l'] *= np.conjugate(dl1)
    b1_truncated_canon['d_l'] *= np.conjugate(dl1)

    def pretty(canon):
        print(angler(np.diag(canon['d_l'])), angler(canon['x']), angler(canon['y']))
    # pretty(b1_truncated_canon)
    # pretty(b2_truncated_canon)

    a[1] = rebuild_from_canon(b1_truncated_canon)
    a[2] = rebuild_from_canon(b2_truncated_canon)

    verify_mub(a)
    c = hadamard_cube(a)
    verify_cube_properties(c)

    # before all the left perm removal and such,
    # reorder_mub(a) was needed to bring the MUB to standard form.
    # after it, hardwired_reorder(a) is enough, a fixed, trivial rearrange.
    # a = reorder_mub(a)
    a = hardwired_reorder(a)

    verify_mub(a)
    c = hadamard_cube(a)
    verify_cube_properties(c)

    x1y1 = (b1_truncated_canon['x'], b1_truncated_canon['y'])
    dl2 = b2_truncated_canon['d_l']
    x2y2 = (b2_truncated_canon['x'], b2_truncated_canon['y'])
    return a, x1y1, dl2, x2y2


mub_type_name = mub_type(a)
if mub_type_name != "Szollosi":
    print("filename", filename, mub_type_name)
    exit()

a = standardize_triplet_order(a)
a, x1y1, dl2, x2y2 = standardize_mub(a)
x1, y1 = x1y1
x2, y2 = x2y2
szollosi = trans(a[1], a[2])

assert filename.startswith("triplets/")
output_filename = filename.replace("triplets/", "szollosis/")

np.save(output_filename, szollosi)

print("filename", filename, "x1", angler(x1), "y1", angler(y1),
    "x2", angler(x2), "y2", angler(y2), "Dl2", angler(np.diag(dl2)))
exit()

c = hadamard_cube(a)
print(np.diag(visualize_clusters(c[0, :, :])))


def test_equivalences(a):
    c = hadamard_cube(a)
    sx = c[0, :, :]
    sy = c[:, 0, :]
    sz = c[:, :, 0]

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
    sxb = Matrix(sx.copy())
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


Wsym = symbols('W')

def symbolic_fourier_basis(x_var, y_var):
    roots = [1, - Wsym ** 2, Wsym, -1, Wsym ** 2, - Wsym]
    b = ones(6, 6)
    for i in range(1, 6, 3):
        for j in range(1, 6, 2):
            b[i, j] *= x_var
    for i in range(2, 6, 3):
        for j in range(1, 6, 2):
            b[i, j] *= y_var
    for i in range(1, 6):
        for j in range(1, 6):
            b[i, j] *= roots[(i * j) % 6]
    return b



# the third element of the triplet is Id.
def symbolic_hadamard_cube(b1, b2):
    a = MutableDenseNDimArray([0] * 108, (3, 6, 6))
    a[0, :, :] = eye(6)
    a[1, :, :] = b1
    a[2, :, :] = b2
    c = MutableDenseNDimArray([0] * 216, (6, 6, 6))
    for i in range(6):
        for j in range(6):
            for k in range(6):
                c[i, j, k] = \
                    Matrix(conjugate(a[0, :, i])).dot(a[1, :, j]) * \
                    Matrix(conjugate(a[1, :, j])).dot(a[2, :, k]) * \
                    Matrix(conjugate(a[2, :, k])).dot(a[0, :, i])
    return ImmutableDenseNDimArray(6 * c)


def simplify_roots(expr):
    e = expr.subs(conjugate(Wsym), Wsym ** 2)
    e = e.subs(Wsym ** 3, 1).subs(Wsym ** 4, Wsym).subs(Wsym ** 5, Wsym ** 2).subs(Wsym ** 6, 1)
    return e


# constraints = extract_constraints_from_symbolic_cube(c)


TP = 2 * np.pi
TIP = 2j * np.pi
W = np.exp(TIP / 3)

x1, y1 = x1y1
x2, y2 = x2y2

x1_sym, y1_sym = symbols('x_1 y_1')
x2_sym, y2_sym = symbols('x_2 y_2')
alpha_sym = [1] + list(symbols('alpha_2:7'))
fourier1_sym = deinterlace(symbolic_fourier_basis(x1_sym, y1_sym) / sqrt(6))
fourier2_sym = deinterlace(symbolic_fourier_basis(x2_sym, y2_sym) / sqrt(6))
fourier2_sym = diag(*alpha_sym) @ fourier2_sym


# not to be confused with c_sym later!
cube_sym = symbolic_hadamard_cube(fourier1_sym, fourier2_sym)
sx = simplify_roots(cube_sym[0, :, :]) * 6
print(Matrix(sx[:3, :3]) - sx[0, 2] * ones(3))

print("====")
print(conjugate_pair_sym(sx))
sx1 = simplify_roots(cube_sym[1, :, :]) * 6
print("---")
print(sx1)

print(angler(conjugate_pair(c[0, :, :])))

print(angler(c[1, :, :]))
exit()


prod = Dagger(fourier1_sym) @ fourier2_sym

flattened = ones(36, 1)
for row in range(6):
    for col in range(6):
        flattened[row * 6 + col] = simplify_roots(prod[row, col]) * 6

print(latex(flattened))
exit()



slicepair_data = collect_slicepair_data(c)
c_sym, sy_sym, phis, variables, slicepair_data_sym = create_symbolic_cube()
x_sym, y_sym = symbols('x y')
fourier_sym = deinterlace(symbolic_fourier_basis(x_sym, y_sym) / sqrt(6))
reconstructed_b1 = evaluate(fourier_sym.subs(x_sym, x1).subs(y_sym, y1).subs(Wsym, W))
assert np.allclose(reconstructed_b1, a[1], atol=1e-4)

for row in range(6):
    for col in range(6):
        c_sym = c_sym.subs(sy_sym[row, col], fourier_sym[row, col])



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


print("Collected constraints:")
for cons in constraints:
    print(latex(Eq(cons, 0)))
