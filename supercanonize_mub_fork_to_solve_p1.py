# input: three lines of canonized_mubs, like
# cat canonized_mubs | grep mub_100.npy | python analyze_canonized_mubs.py

import sys
import numpy as np
import matplotlib.pyplot as plt

# filename normalized/mub_100.npy i 1 D_l [  -0.      -26.7955 -173.147   -38.4863 -168.1707 -150.0384] P_l [0 5 2 3 1 4] x 55.11879687165239 y 0.0005514574849221608 P_r [4 1 2 3 0 5] D_r [0. 0. 0. 0. 0. 0.] distance 4.003481295679986e-06


TP = 2 * np.pi
TIP = 2j * np.pi
W = np.exp(TIP / 3)
WW = np.exp(TIP / 12)
BOARD = [(i, j) for i in range(6) for j in range(6)]


def degrees_to_phases(d):
    return np.exp(1j * np.pi / 180 * d)


def perm_to_pm(p):
    return np.eye(6)[p, :]


# TODO build a lib, do not copypaste this function everywhere.
# x and y are phases. this is exactly formula (3) in https://arxiv.org/pdf/0902.0882.pdf
# not sure about numerical precision when doing (sixth ** 5) instead of -W.
def canonical_fourier(x, y):
    ws = np.ones((6, 6), dtype=np.complex128)
    ws[1:6:3, 1:6:2] *= x
    ws[2:6:3, 1:6:2] *= y
    sixth = - np.conjugate(W)
    for i in range(1, 6):
        for j in range(1, 6):
            ws[i, j] *= sixth ** ((i * j) % 6)
    return ws


def factors_to_basis(factors):
    left_phases, left_pm, x, y, right_pm, right_phases = factors
    return np.diag(left_phases) @ left_pm @ canonical_fourier(x, y) @ right_pm @ np.diag(right_phases)


def serialized_to_factors(serialized_basis):
    left_degrees, left_permutation, x_degrees, y_degrees, right_permutation, right_degrees = serialized_basis
    x = degrees_to_phases(x_degrees)
    y = degrees_to_phases(y_degrees)
    left_phases = degrees_to_phases(left_degrees)
    right_phases = degrees_to_phases(right_degrees)

    # beware of the T here!!! left and right permutation matrices
    # are represented differently as permutations.
    left_pm = perm_to_pm(left_permutation).T
    right_pm = perm_to_pm(right_permutation)

    factors = (left_phases, left_pm, x, y, right_pm, right_phases)
    basis = factors_to_basis(factors)
    assert np.allclose(np.conjugate(basis.T) @ basis, 6 * np.eye(6)), "does not seem to be a basis: " + str(np.conjugate(basis.T) @ basis)
    assert np.allclose(np.abs(basis), 1), "does not seem to be Hadamard: " + str(np.abs(basis))
    return factors


mubs = {}
for l in sys.stdin:
    l = l.replace("[", "").replace("]", "")
    a = l.strip().split()
    assert len(a) == 38
    filename = a[1]
    basis = int(a[3])
    left_degrees = np.array(list(map(float, a[5: 5 + 6])))
    right_degrees = np.array(list(map(float, a[30: 30 + 6])))
    left_permutation = np.array(list(map(int, a[12: 12 + 6])))
    right_permutation = np.array(list(map(int, a[23: 23 + 6])))
    assert a[18] == "x"
    x_degrees = float(a[19])
    y_degrees = float(a[21])
    if filename not in mubs:
        mubs[filename] = []
    factors = serialized_to_factors((left_degrees, left_permutation, x_degrees, y_degrees, right_permutation, right_degrees))
    mubs[filename].append(factors)


'''
assert len(mubs.keys()) == 2
keys = sorted(mubs.keys())
factorized_mub_m = mubs[keys[0]]
factorized_mub_n = mubs[keys[1]]
print("qMUB M:", keys[0])
print("qMUB N:", keys[1])
assert len(factorized_mub_m) == len(factorized_mub_n)== 3
'''

assert len(mubs.keys()) == 1
factorized_mub = mubs[list(mubs.keys())[0]]


def remove_right_effects(factors):
    left_phases, left_pm, x, y, right_pm, right_phases = factors
    return left_phases, left_pm, x, y, np.eye(6), np.ones(6)


def remove_right_effects_from_mub(factorized_mub):
    factorized_mub2 = []
    for i in range(3):
        factorized_mub2.append(remove_right_effects(factorized_mub[i]))
    return factorized_mub2


def reorder_by_fourier_params(factorized_mub):
    def signature(x, y):
        # when this signature is in increasing order, we have (a, a), (a, 0), (-a, 0).
        return - np.angle(x) - np.angle(y)

        # when this signature is in increasing order, we have (a, 0), (-a, 0), (a, a).
        # that's the order of normalized/mub_120.npy
        return - np.angle(x) + 10 * np.angle(y)

    signatures = [signature(factors[2], factors[3]) for factors in factorized_mub]

    # signatures = [- np.angle(factors[2]) + 10 * np.angle(factors[3]) for factors in factorized_mub]


    perm = np.argsort(np.array(signatures))
    factorized_mub2 = [None, None, None]
    for to_pos, from_pos in enumerate(perm):
        factorized_mub2[to_pos] = factorized_mub[from_pos]

    # verify
    signatures2 = [signature(factors[2], factors[3]) for factors in factorized_mub2]
    perm = np.argsort(np.array(signatures2))
    assert perm.tolist() == list(range(3))

    return factorized_mub2


def switch_perm_phase(pm, phases):
    matrix = pm @ np.diag(phases)
    phases2 = np.sum(matrix, axis=1)
    matrix2 = np.diag(1 / phases2) @ matrix
    assert np.allclose(matrix, np.diag(phases2) @ matrix2)
    assert np.allclose(matrix2 * (matrix2 - 1), 0)
    pm2 = np.around(matrix2).real.astype(int)
    return phases2, pm2


def switch_phase_perm(phases, pm):
    phases2, pm2 = switch_perm_phase(pm.T, phases)
    return pm2.T, phases


def apply_left_phase(factorized_mub, left_phases_to_apply):
    factorized_mub2 = []
    for i in range(3):
        left_phases, left_pm, x, y, right_pm, right_phases = factorized_mub[i]
        left_phases_after = left_phases * left_phases_to_apply
        factorized_mub2.append((left_phases_after, left_pm, x, y, right_pm, right_phases))
    return factorized_mub2


def apply_left_permutation(factorized_mub, left_pm_to_apply):
    factorized_mub2 = []
    for i in range(3):
        left_phases, left_pm, x, y, right_pm, right_phases = factorized_mub[i]
        left_phases_after, pm2 = switch_perm_phase(left_pm_to_apply, left_phases)
        factorized_mub2.append((left_phases_after, pm2 @ left_pm, x, y, right_pm, right_phases))
    return factorized_mub2


def remove_first_left_effect(factorized_mub):
    factors_0 = factorized_mub[0]
    left_phases_0, left_pm_0 = factors_0[:2]
    factorized_mub = apply_left_phase(factorized_mub, 1 / left_phases_0)
    factorized_mub = apply_left_permutation(factorized_mub, left_pm_0.T)
    return factorized_mub


factorized_mub = remove_right_effects_from_mub(factorized_mub)
factorized_mub = reorder_by_fourier_params(factorized_mub)
factorized_mub = remove_first_left_effect(factorized_mub)


def dump_basis(factors):
    left_phases, left_pm, x, y, right_pm, right_phases = factors
    # left_pm, left_phases = switch_phase_perm(left_phases, left_pm)
    print(np.angle(left_phases) * 180 / np.pi)
    print(left_pm.astype(int))
    print(np.angle(x) * 180 / np.pi, np.angle(y) * 180 / np.pi)


def dump_mub(factorized_mub):
    for i in range(3):
        print("=====")
        print(i)
        dump_basis(factorized_mub[i])


def short_dump_mub(factorized_mub):
    # B_1 is F(a,a), we don't dump that.
    np.set_printoptions(precision=12, suppress=True, linewidth=100000)
    def angler(x):
        return " ".join(map(str, (np.angle(x) * 180 / np.pi).tolist()))
    perm1 = np.argmax(factorized_mub[1][1], axis=1)
    perm2 = np.argmax(factorized_mub[2][1], axis=1)
    print(angler(factorized_mub[1][0]), perm1, angler(factorized_mub[2][0]), perm2)


def angler(x):
    return np.angle(x) * 180 / np.pi


short_dump_mub(factorized_mub)


mub = []
for i in range(3):
    basis = factors_to_basis(factorized_mub[i])
    mub.append(basis)
mub = np.array(mub)


def closeness(a, b):
    return np.sum(np.abs(a - b) ** 2)


# numpy code reimplementing the tf code search.py:loss_fn()
def loss_function(mub):
    terms = []
    for u in mub:
        prod = np.conjugate(u.T) @ u / 6
        terms.append(closeness(prod, np.eye(6)))

    target = 1 / 6 ** 0.5
    for i in range(3):
        for j in range(i + 1, 3):
            # / 6 because what we call basis is 6**2 times what search.py calls basis.
            prod = np.conjugate(mub[i].T) @ mub[j] / 6
            terms.append(closeness(np.abs(prod), target))
    return sum(terms)


# (B_i^dag B_j)_kl =.
def prod_elem(i, j, k, l):
    aprod = np.conjugate(mub[i].T) @ mub[j]
    return aprod[k, l]


def product_angles():
    normalized = []
    for i in (0, ):
        for j in (1, 2):
            for k in (4, ):
                for l in range(6):
                    p = prod_elem(i, j, k, l) / 6
                    rounded_abs = (np.abs(p) * 1000).astype(int)
                    assert rounded_abs in (355, 385, 425)
                    rounded_abs /= 1000
                    angle = np.angle(p)
                    # 10^8 needed to avoid spurious coincidences because of rounding,
                    # but to still avoid spurious differences because of precision errors.
                    rounded_angle = int(angle * 10 ** 8)
                    rounded_angle /= 10 ** 8
                    rounded_angle *= 180 / np.pi
                    normalized.append((i, j, k, l, rounded_abs, rounded_angle))
                    print(i, j, k, l, rounded_abs, rounded_angle)
    normalized = np.array(normalized)
    return normalized


# product_angles() ; exit()

def deconstruct_product(prod):
    sub0 = prod[:2, :2].copy()
    sub0 /= np.abs(sub0)
    sub1 = prod[:2, 2:4].copy()
    sub1 /= np.abs(sub1)
    sub2 = prod[:2, 4:].copy()
    sub2 /= np.abs(sub2)
    print(np.around(angler(sub1 / sub0)))
    print(np.around(angler(sub2 / sub0)))


def inspect_fprods():
    x = factorized_mub[0][2]
    F = canonical_fourier(x, 1)
    for (i, j) in [(0, 1), (1, 2), (2, 0)]:
        prod = np.conjugate(mub[i].T) @ mub[j]
        # fprod is a matrix of the form D @ P.
        fprod = F @ prod @ np.conjugate(F.T) / 36
        # moreover, the elements of D are symmetric to a line. let's move that to 1.
        # after this normalization, it seems like there are only finitely many variations for angles across all the qmubs.
        direction = fprod.sum()
        direction /= np.abs(direction)
        fprod /= direction
        vecs = fprod.flatten()
        vecs = vecs[np.abs(vecs) > 0.5]
        angles = np.sort(np.angle(vecs))
        print("*", i, j, angles / np.pi * 180)

        # plt.scatter(vecs.real, vecs.imag)

    # plt.show()


# inspect_fprods() ; exit()


np.set_printoptions(precision=3, suppress=True, linewidth=100000)
for (i, j) in [(0, 1), (1, 2), (2, 0)]:
    prod = np.conjugate(mub[i].T) @ mub[j]
    # deconstruct_product(prod)
    print(f"| B_{i}^dagger B_{j} | / 6")
    print(np.abs(prod) / 6)
    ang = angler(prod)
    print(angler(prod))
    # these three elements determine the product up to combinatorics,
    # but they are only two degrees of freedom really, because
    # ang[0,1] = ang[0,0] + delta, ang[1,0] = ang[0,0] - delta, for some delta.
    for (i, j) in [(0, 0), (0, 1), (1, 0)]:
        print(i, j)
        print(ang - ang[i, j])


def dump_products(mub):
    np.set_printoptions(precision=4, suppress=True, linewidth=100000)
    for i in range(3):
        prod = np.conjugate(mub[i].T) @ mub[i]
        print(f"| B_{i}^dagger B_{i} | / 6")
        print(np.abs(prod) / 6)

    for (i, j) in [(0, 1), (1, 2), (2, 0)]:
        prod = np.conjugate(mub[i].T) @ mub[j]
        print(f"| B_{i}^dagger B_{j} | / 6")
        print(np.abs(prod) / 6)


# dump_products(mub)


# also scale it back to be actually unitary.
def put_back_id(mub):
    return np.concatenate([np.eye(6, dtype=np.complex128)[np.newaxis, ...], mub / 6 ** 0.5])


loss = loss_function(mub)
print("loss", loss)

np.save('tmp.npy', put_back_id(mub))
print("normalized qMUB saved into tmp.npy")

import sympy
from sympy import symbols, Matrix, Transpose, conjugate, sqrt, cbrt, Rational
from sympy import expand, factor, cancel, nsimplify, simplify, fraction
from sympy.physics.quantum.dagger import Dagger
from sympy.matrices.dense import matrix_multiply_elementwise


# symbol of third root of unity.
Wsym = symbols('W')


def simplify_roots(expr):
    e = expr.subs(conjugate(Wsym), Wsym ** 2)
    e = e.subs(Wsym ** 3, 1).subs(Wsym ** 4, Wsym).subs(Wsym ** 5, Wsym ** 2).subs(Wsym ** 6, 1)
    return e


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


def symbolic_fourier_basis(x_var, y_var):
    roots = [1, - Wsym ** 2, Wsym, -1, Wsym ** 2, - Wsym]
    b = sympy.ones(6, 6)
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


# we don't do right phase, it is identity in the cases we care about.
def symbolic_generalized_fourier_basis(left_phases_var, left_pm, x_var, y_var):
    fourier = symbolic_fourier_basis(x_var, y_var)
    return sympy.diag(left_phases_var, unpack=True) @ left_pm.astype(int) @ fourier


# indx is needed to create the variables.
# the only information used from factors is the left_pm.
# we assume that all of them are F(x, 1) bases.
def create_basis(indx, factors):
    left_phases, left_pm, x, y, right_pm, right_phases = factors
    assert np.isclose(np.angle(x) / TP * 360, 55.118, atol=1e-4)
    assert np.isclose(y, 1)
    left_phases_var = [sympy.symbols(f'p{indx}{i}') for i in range(6)]
    x_var = sympy.symbols('x')
    return left_phases_var, symbolic_generalized_fourier_basis(left_phases_var, left_pm, x_var, 1)


# all the below is true for normalized/mub_10024.npy and not much else.
# the goal is to reverse engineer this single one. the rest is combinatorics,
# the manifold is zero dimensional.



A = 0.42593834 # new name W
B = 0.35506058 # new name U
C = 0.38501704 # new name V
D = 0.40824829
numerical_magic_constants = [36 * c **2 for c in [A, B, C]]

WWsym = sympy.symbols('phi') # 12th root of unity.
magnitudes_sym = sympy.symbols('A B C')
xvar = symbols('x')
alpha01sym, delta01sym = sympy.symbols('alpha1 delta1')
alpha02sym, delta02sym = sympy.symbols('alpha2 delta2')
left_phases_var_1 = Matrix([[sympy.symbols(f'p1{i}') for i in range(6)]]) # row vector
left_phases_var_2 = Matrix([[sympy.symbols(f'p2{i}') for i in range(6)]])

d_1 = factorized_mub[1][0]
d_2 = factorized_mub[2][0]
x = factorized_mub[0][2]
alpha01 = 0.8078018037463457+0.589454193185654j
delta01 = -0.12834049204788406+0.9917301639563593j
alpha02 = 0.38501828415482425+0.9229089450571356j
delta02 = 0.923033846322939-0.3847187525222561j


# between -30 and 30 actually, so that 60-divisible angles
# stably land around 0.
def rot_to_60(p_orig):
    p = p_orig
    while not(-np.pi / 6 <= np.angle(p) < np.pi / 6):
        p *= -np.conjugate(W)
    return p



def identify_alpha_delta(b0, b1):
    prod = np.conjugate(b0.T) @ b1
    msquared = np.abs(prod) ** 2
    normed = prod.copy()
    normed /= np.abs(normed)

    magnitude_masks = [np.isclose(msquared, numerical_magic_constants[i], atol=1e-4).astype(int) for i in range(3)]
    magnitude_masks = np.array(magnitude_masks)

    A_mask = magnitude_masks[0]
    collection = []
    for (i, j) in BOARD:
        if A_mask[i, j]:
            rot = normed[i, j]
            collection.append(rot)
    collection = np.array(collection)

    coll_60 = []
    epsilon = 0.01
    for rot in collection:
        while not(epsilon <= np.angle(rot) < np.pi / 3 + epsilon):
            rot *= np.exp(1j * np.pi / 3)
        coll_60.append(rot)
    coll_60 = np.array(coll_60)
    # -> coll_60 is supposed to have only 3 distinct values up to numerical precision.
    #    12 alpha, 6 alpha delta, 6 alpha conj(delta).
    pairs = coll_60[None, :] / coll_60[:, None]
    neighbors = np.isclose(pairs, 1, atol=1e-4)
    neighbor_counts = neighbors.sum(axis=1)
    # we are interested in the ones that have exactly 12 neighbors:
    alpha_circle = collection[neighbor_counts == 12]
    # plt.scatter(collection.real, collection.imag) plt.show()
    assert len(alpha_circle) == 12

    # any element of alpha_circle will do as alpha, it's fully symmetric
    # we break tie by choosing the one closest to 0 degrees.
    dist_to_0 = np.abs(np.angle(alpha_circle))
    i = np.argmin(dist_to_0)
    alpha = alpha_circle[i]
    collection /= alpha
    alpha_circle = collection[neighbor_counts == 12]

    # more exactly delta union conj(delta) circle:
    delta_circle = collection[neighbor_counts == 6]
    assert len(delta_circle) == 12
    # after the previous division by alpha,
    # any element of the delta_circle will do as delta.
    # we break the tie by choosing the one closest to 0,
    # that's supposed to be either ~7.3737 degrees or ~60-7.3737 degrees.
    # we then mirror it to ~7.3737.
    dist_to_0 = np.abs(np.angle(delta_circle))
    i = np.argmin(dist_to_0)
    delta = delta_circle[i]
    alpha = rot_to_60(alpha)
    delta = rot_to_60(delta)
    if angler(delta) < 0:
        delta = np.conjugate(delta)
        if angler(delta) > 15:
            delta = WW * np.conjugate(delta)
            assert angler(delta) < 15
    print("alpha", angler(alpha), "delta", angler(delta))
    return alpha, delta


# BEWARE: the result is yet to be multiplied by 6 * alphasym.
def reconstruct_product(b0, b1, alpha, delta, alphasym, deltasym):
    prod = np.conjugate(b0.T) @ b1
    normed = prod.copy()
    normed /= np.abs(normed)

    roots = WW ** np.arange(12)
    candidates = np.array([1, delta, 1 / delta])
    all_candidates = alpha * candidates[:, None] * roots[None, :]

    masks = [np.isclose(normed, all_candidates[i, j], atol=1e-5).astype(int) for i in range(3) for j in range(12)]
    masks = np.array(masks).reshape((3, 12, 6, 6))
    assert masks.sum() == 36, "some elements of the product could not be identified"
    # TODO we can do it without increasing the degree
    roots_sym = Matrix([WWsym ** i for i in range(12)])
    candidates_sym = [1, deltasym, conjugate(deltasym)]

    normed_prod_sym = sympy.ones(6, 6)
    for i in range(6):
        for j in range(6):
            onehot = masks[:, :, i, j]
            element_index = np.argmax(onehot.sum(axis=1))
            rotation = np.argmax(onehot.sum(axis=0))
            assert onehot[element_index, rotation] == 1
            normed_prod_sym[i, j] *= candidates_sym[element_index] * roots_sym[rotation]
    # assert np.allclose(sym_to_num(alphasym * normed_prod_sym), normed)

    msquared = np.abs(prod) ** 2
    masks = [np.isclose(msquared, numerical_magic_constants[i]).astype(int) for i in range(3)]
    masks = np.array(masks)
    magnitude_matrix_sym = sympy.ones(6, 6)
    for i in range(6):
        for j in range(6):
            element_index = np.argmax(masks[:, i, j])
            magnitude_matrix_sym[i, j] *= magnitudes_sym[element_index]
    # assert np.allclose(sym_to_num(6 * magnitude_matrix_sym), np.abs(prod))
    final_result = matrix_multiply_elementwise(normed_prod_sym, magnitude_matrix_sym)
    # assert np.allclose(sym_to_num(6 * alphasym * final_result), prod)
    return final_result


def reconstruct_product_full_service(b0, b1, indx):
    alpha, delta = identify_alpha_delta(b0, b1)
    alphasym, deltasym = sympy.symbols(f"alpha{indx}, delta")
    p = reconstruct_product(b0, b1, alpha, delta, alphasym, deltasym)
    return p


def alpha_removal(mub):
    alpha1, delta1 = identify_alpha_delta(mub[0], mub[1])
    mub[1] /= alpha1
    alpha2, delta2 = identify_alpha_delta(mub[0], mub[2])
    mub[2] /= alpha2

    alpha1, delta1 = identify_alpha_delta(mub[0], mub[1])
    assert np.isclose(alpha1 ** 6, 1, atol=1e-3)
    alpha2, delta2 = identify_alpha_delta(mub[0], mub[2])
    assert np.isclose(alpha2 ** 6, 1, atol=1e-3)

    alpha0, delta0 = identify_alpha_delta(mub[1], mub[2])
    assert np.isclose(alpha0 ** 6, 1, atol=1e-3)
    return mub


def sym_to_num(formula):
    f = formula.subs(Wsym, W).subs(WWsym, np.exp(TIP/12))
    f = f.subs(xvar, x)
    # f = f.subs(BperAvar, B/A).subs(CperAvar, C/A)
    f = f.subs(magnitudes_sym[0], A).subs(magnitudes_sym[1], B).subs(magnitudes_sym[2], C)
    f = f.subs(alpha01sym, alpha01).subs(delta01sym, delta01)
    f = f.subs(alpha02sym, alpha02).subs(delta02sym, delta02)
    for i in range(6):
        f = f.subs(left_phases_var_1[i], d_1[i])
        f = f.subs(left_phases_var_2[i], d_2[i])
    try:
        a = np.array(f, dtype=np.complex128)
    except:
        print("failed to fully evaluate", formula)
        print("still variables left in", f)
        raise
    return np.squeeze(a)


def create_symbolic_mub(factorized_mub):
    symbolic_bases = []
    for indx in range(3):
        factors = factorized_mub[indx]
        left_phases_var, b = create_basis(indx, factors)
        if indx == 0:
            for v in [sympy.symbols(f'p{indx}{i}') for i in range(6)]:
                b = b.subs(v, 1)
        symbolic_bases.append(b)
    return symbolic_bases


mub = alpha_removal(mub)

alpha, delta = identify_alpha_delta(mub[0], mub[1])
prod01reconstructed_sym = reconstruct_product(mub[0], mub[1], alpha, delta, alpha01sym, delta01sym)
print("B_1^\\dag B_2 = 6 \\alpha_2", sympy.latex(prod01reconstructed_sym))
prod01reconstructed_sym *= 6 * alpha01sym # it immediately executes, so it's nicer before it

alpha, delta = identify_alpha_delta(mub[0], mub[2])
prod02reconstructed_sym =  reconstruct_product(mub[0], mub[2], alpha, delta, alpha02sym, delta02sym)
print("B_1^\\dag B_3 = 6 \\alpha_3", sympy.latex(prod02reconstructed_sym))
prod02reconstructed_sym *= 6 * alpha02sym


symbolic_bases = create_symbolic_mub(factorized_mub)

# why range(2) not range(3)?
# we don't do anything with B_2 (the last basis) now,
# so we don't even substitute its phases.
# TODO we should.
for i in range(3):
    assert np.allclose(sym_to_num(symbolic_bases[i]), mub[i])

# this is where we collect the D_1 elements written up in terms of A,B,C,x,W,WW.
phase_solution = left_phases_var_1[:]

symbolic_bases[1] = symbolic_bases[1].subs(left_phases_var_1[0], conjugate(xvar))
phase_solution[0] = conjugate(xvar)
assert np.allclose(sym_to_num(symbolic_bases[1]), mub[1])
symbolic_bases[1] = symbolic_bases[1].subs(left_phases_var_1[1], -Wsym ** 2)
phase_solution[1] = -Wsym ** 2
assert np.allclose(sym_to_num(symbolic_bases[1]), mub[1])
symbolic_bases[1] = symbolic_bases[1].subs(left_phases_var_1[3], -Wsym * conjugate(xvar) * left_phases_var_1[4])
phase_solution[3] = -Wsym * conjugate(xvar) * left_phases_var_1[4]
assert np.allclose(sym_to_num(symbolic_bases[1]), mub[1])
symbolic_bases[1] = simplify_roots(enforce_norm_one(symbolic_bases[1], [xvar]))

print("B_1", symbolic_bases[1])

# we know less about symbolic_bases[2], but we know a few things:
assert np.allclose(sym_to_num(symbolic_bases[2]), mub[2])
symbolic_bases[2] = symbolic_bases[2].subs(left_phases_var_2[0], -1)
symbolic_bases[2] = symbolic_bases[2].subs(left_phases_var_2[2], left_phases_var_1[2])
symbolic_bases[2] = simplify_roots(enforce_norm_one(symbolic_bases[2], [xvar]))
assert np.allclose(sym_to_num(symbolic_bases[2]), mub[2])


prod01sym = Dagger(symbolic_bases[0]) @ symbolic_bases[1]
prod01sym = simplify_roots(prod01sym)
print("product01", prod01sym)

prod02sym = Dagger(symbolic_bases[0]) @ symbolic_bases[2]
prod02sym = simplify_roots(prod02sym)
print("product02", prod02sym)

assert np.allclose(sym_to_num(prod01sym), np.conjugate(mub[0].T) @ mub[1])
assert np.allclose(sym_to_num(prod02sym), np.conjugate(mub[0].T) @ mub[2])



def subs_roots(f):
    f = f.subs(Wsym, - Rational(1, 2) + sympy.I * sympy.sqrt(3) * Rational(1, 2))
    f = f.subs(WWsym, Rational(1, 2) * sympy.I + sympy.sqrt(3) * Rational(1, 2))
    f = expand(f)
    return f


F = symbolic_fourier_basis(xvar, 1)
fprod = F * prod01reconstructed_sym * Dagger(F) * Rational(1, 36)
fprod = subs_roots(fprod)
print(expand(fprod / alpha01sym), sym_to_num(fprod / alpha01sym))

'''
vecs = sym_to_num(fprod / alphasym).flatten()
plt.scatter(vecs.real, vecs.imag)
plt.show()
'''


diff01 = prod01sym - prod01reconstructed_sym
assert np.allclose(sym_to_num(diff01), 0, atol=1e-4)
diff01 = subs_roots(diff01)
assert np.allclose(sym_to_num(diff01), 0, atol=1e-4)

diff02 = prod02sym - prod02reconstructed_sym
assert np.allclose(sym_to_num(diff02), 0, atol=1e-4)
diff02 = subs_roots(diff02)
assert np.allclose(sym_to_num(diff02), 0, atol=1e-4)


'''
for i in range(6):
    for j in range(6):
        print(i, j, "=>", diff[i, j])
'''


def extract_directions(eq, variables):
    bias = eq
    for v in variables:
        bias = bias.subs(v, 0)
    return Matrix([sympy.diff(eq, v) for v in variables]), bias


positions01 = [ #  BFE
    (0, 0), # nicest of the 6 As, tells about A alpha
    (3, 3), # nicest of the 6 Cs, tells about A alpha
#    (2, 2), # nicest of the 6 Es, tells about C alpha
#    (0, 1), # nicest of the 3 Bs, tells about B alpha delta
#    (1, 0), # nicest of the 3 Fs, tells about B alpha conj(delta)
    (2, 3), # nicest of the 2x3 Ds, tells about A alpha delta
    (3, 2), # nicest of the 2x3 Gs, tells about A alpha conj(delta)
]

# positions02 = [ (i, j) for i in range(6) for j in range(6) ]
positions02 = [(0, 0), (0, 2), (0, 3), (0, 5), (1, 2)]


# variables = [sympy.symbols(f'p1{i}') for i in range(6)]
# the remaining variables after the substitutions above:
variables1 = sympy.symbols('p12 p14 p15 A')
variables2 = sympy.symbols('p21 p23 p24 p25 A')

# we cannot do both yet, but at least let's do either.
which_to_solve = 2
if which_to_solve == 1:
    variables = variables1
    diff = diff01
    positions = positions01
    alphasym = alpha01sym
if which_to_solve == 2:
    variables = variables2
    diff = diff02
    positions = positions02
    alphasym = alpha02sym
else:
    assert False




def show_points(points, title):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.scatter(points.real, points.imag)
    partition = "ABCDED FAGEGC CDEDAB GEGCFA EDABCD GCFAGE".replace(" ", "")
    if False:
        for i, clss in enumerate(partition):
            ax.annotate(" " * (i // 4) + f"{i}-{clss}", (points.real[i], points.imag[i]))
    plt.title(title)
    plt.show()


def show_directions(directions, biases):
    directions_num = sym_to_num(directions)
    biases_num = sym_to_num(biases)
    for i in range(6):
        points = directions_num[:, i]
        show_points(points, title=str(variables[i]))
    points = biases_num
    show_points(points, title="biases")


# show_directions(all_directions, all_biases) ; exit()



# positions = [(0, i) for i in range(4)] # + [(1, 0)]


def collect_linear_system(diff, variables, positions):
    directions = []
    biases = []
    # note: if we keep the first row and the first element of the second row.
    # that's already rank 6, enough to determine all our variables.
    for (i, j) in positions:
        direction, bias = extract_directions(diff[i, j], variables)
        directions.append(direction)
        biases.append(bias)

    # a python list of column vector sympy Matrices is not trivial to convert to a sympy Matrix:
    directions = Matrix([[coeff[0] for coeff in direction.tolist()] for direction in directions])
    biases = Matrix(biases)
    return directions, biases


directions, biases = collect_linear_system(diff, variables, positions)

# show_directions(directions, biases) ; exit()

directions = expand(directions)
biases = expand(biases)

print("---")

def dump_eq(variable, eq):
    if isinstance(variable, str):
        name = variable
    else:
        name = sympy.latex(variable)
    print("\\begin{equation}")
    print(name, "=", sympy.latex(nsimplify(eq)))
    print("\\end{equation}")
    print()


dump_eq("A", directions)
dump_eq("x", variables)
dump_eq("b", biases)


def numerically_solve_linear_system(directions, biases):
    directions_num = sym_to_num(directions)
    biases_num = sym_to_num(biases)

    np.set_printoptions(precision=12, suppress=False, linewidth=100000)
    u, s, vh = np.linalg.svd(directions_num)
    print("singular values:", s)

    predictions = np.linalg.lstsq(directions_num, -biases_num, rcond=None)[0]
    return predictions


predictions = numerically_solve_linear_system(directions, biases)
expectations = sym_to_num(Matrix(variables))
assert np.allclose(predictions, expectations, atol=1e-4)
print("passed the test:", Matrix(variables), "can be reconstructed from these elements of the product")


def create_and_verify_eq(formula):
    eq = subs_roots(formula)
    num = sym_to_num(eq)
    print(eq, num)
    try:
        assert np.isclose(num, 0, atol=1e-5)
    except:
        print(eq, "should be numerically 0 but is", num)
    return eq

print("----")


from sympy.solvers.solveset import linsolve


# predicting variables aka (p12, p14, p15, A, B, C)
predictions_sym = linsolve((directions, -biases), variables)
# take single element of FiniteSet. it's a tuple, but we convert it to list:
predictions_sym = list(list(predictions_sym)[0])

for i in range(len(predictions_sym)):
    predictions_sym[i] = simplify(factor(simplify(predictions_sym[i]), gaussian=True))

predictions_sym = Matrix(predictions_sym)

assert np.allclose(sym_to_num(predictions_sym), expectations, atol=1e-4)

def mytogether(a):
    a = sympy.polys.rationaltools.together(a)
    a = sympy.polys.polytools.factor(a, gaussian=True)
    return a

for variable, prediction in zip(variables, predictions_sym):
    prediction = enforce_norm_one(prediction, [xvar])
    prediction = mytogether(prediction)
    dump_eq(variable, prediction)


# TODO this is prod01 specific
phase_solution[2] = predictions_sym[0]
phase_solution[4] = predictions_sym[1]
phase_solution[5] = predictions_sym[2]
phase_solution[1] = subs_roots(phase_solution[1])
# phase_solution[3] still has some p14 in it, let's get rid of it. also W:
phase_solution[3] = subs_roots(expand(phase_solution[3].subs(left_phases_var_1[4], phase_solution[4])))




# turning a/c + b/c into (a+b)/c, much much shorter:
for i in range(6):
    phase_solution[i] = sympy.polys.rationaltools.together(phase_solution[i])
    phase_solution[i] = sympy.polys.polytools.factor(phase_solution[i], gaussian=True)
    # phase_solution[i] = sympy.polys.polytools.cancel(phase_solution[i])

phase_solution = Matrix(phase_solution)

assert np.allclose(sym_to_num(phase_solution), d_1, atol=1e-4)

print("congratulations! D_1 (hence B_1) and A completely written up in terms of alpha, delta, x. B, C could be written up, if needed.")



exit()



A = 0.42593834 # new name W
B = 0.35506058 # new name U
C = 0.38501704 # new name V


print("++++++++")

A_sym, B_sym, C_sym = magnitudes_sym

print(sym_to_num(4 * A_sym ** 2 + B_sym ** 2 + C_sym ** 2), "supposed to be 1.")
print(sym_to_num(2 * A_sym ** 2 - B_sym * (B_sym + sqrt(3) * C_sym)), "supposed to be 0.")

B_solved = sqrt(3) * Rational(1, 4) - sqrt(3 - 16 * A_sym ** 2) / 4
C_solved = 1 - sqrt(3) * B_solved

assert np.isclose(sym_to_num(B_solved), sym_to_num(B_sym))
assert np.isclose(sym_to_num(C_solved), sym_to_num(C_sym))

# useful intermediary variable. (4/sqrt(3)*A_sym, Q_solved) is on the unit circle.
Q_solved = sqrt(1 - A_sym ** 2 * 16 / 3)
assert np.isclose(sym_to_num(B_sym), sym_to_num(sqrt(3) / 4 * (1 - Q_solved)))

# This is Nayral et al's loss, not mine. TODO try that one, and others.
# maybe there's a serendipitously one good among the many equivalent loss functions.
loss_nayral_sym = 6 * (4 * (A_sym ** 2 - Rational(1, 6)) ** 2 + (B_sym ** 2 - Rational(1, 6)) ** 2 + (C_sym ** 2 - Rational(1, 6)) ** 2)

loss_nayral_in_terms_of_A_sym = expand(loss_nayral_sym.subs(B_sym, B_solved).subs(C_sym, C_solved))
print("+++++")
print(loss_nayral_in_terms_of_A_sym)
# -> 84*A**4 - 3*sqrt(3)*A**2*sqrt(3 - 16*A**2) - 36*A**2 + 3*sqrt(3)*sqrt(3 - 16*A**2)/8 + 31/8

A2_sym = sympy.symbols('A2') # A squared
loss_nayral_in_terms_of_A2_sym = 84*A2_sym**2 - 3*sqrt(3)*A2_sym*sqrt(3 - 16*A2_sym) - 36*A2_sym + 3*sqrt(3)*sqrt(3 - 16*A2_sym)/8 + Rational(31, 8)
print("loss_nayral_in_terms_of_A_sym", sym_to_num(loss_nayral_in_terms_of_A_sym))
print("loss_nayral_in_terms_of_A2_sym", sym_to_num(loss_nayral_in_terms_of_A2_sym.subs(A2_sym, A ** 2)))

Wa = predictions_sym[-1] / 6
print("Wa", sym_to_num(Wa), np.abs(sym_to_num(Wa)))
loss_nayral_in_terms_of_x_and_delta_sym = expand(loss_nayral_in_terms_of_A2_sym.subs(A2_sym, Wa * conjugate(Wa)))
loss_nayral_in_terms_of_x_and_delta_sym = enforce_norm_one(loss_nayral_in_terms_of_x_and_delta_sym, [xvar, delta01sym])
print("loss_nayral_in_terms_of_x_and_delta_sym")
print(loss_nayral_in_terms_of_x_and_delta_sym)
print(sym_to_num(loss_nayral_in_terms_of_x_and_delta_sym))

# that subs is tricky. xvar is a constrained to be a phase, so Derivative(conjugate(xvar), xvar) = - conjugate(xvar) ** 2,
# which would not be true in general, so diff cannot deduce it without our help.
dloss_dx = sympy.diff(loss_nayral_in_terms_of_x_and_delta_sym, xvar).subs(sympy.Derivative(conjugate(xvar), xvar), - conjugate(xvar) ** 2)
dloss_ddelta = sympy.diff(loss_nayral_in_terms_of_x_and_delta_sym, delta01sym).subs(sympy.Derivative(conjugate(delta01sym), delta01sym), - conjugate(delta01sym) ** 2)

# TODO that fails the important test that it should be zero at (x, delta).
print(dloss_dx, sym_to_num(dloss_dx))
print(dloss_ddelta, sym_to_num(dloss_ddelta))


from sympy.utilities.lambdify import lambdify

mapper = lambdify([xvar, delta01sym], loss_nayral_in_terms_of_x_and_delta_sym, "numpy")
circle = np.exp(TIP * np.linspace(0, 1, 300))
torus_x, torus_delta = np.meshgrid(circle, circle)
grid = mapper(torus_x, torus_delta)
# TODO this it supposed to be 0 and it's not.
plt.imshow(np.imag(grid))
plt.show()

plt.imshow(np.real(grid))
plt.contour(np.real(grid))
plt.show()

plt.imshow(np.real(lambdify([xvar, delta01sym], dloss_dx, "numpy")(torus_x, torus_delta)))
plt.show()


# sqrt((19/98 - 13/(98 2^(1/3) (49 sqrt(3) - 53)^(1/3)) + (49 sqrt(3) - 53)^(1/3)/(98 2^(2/3))))
weird_const = cbrt(49 * sqrt(3) - 53)
supposed_to_be_A = sqrt((Rational(19, 98) - Rational(13, 1) / (98 * cbrt(2) * weird_const) + weird_const / (98 * cbrt(4))))
print(sym_to_num(supposed_to_be_A))
print("_________")
exit()



def dump_solutions_in_python():
    for i in range(6):
        numer, denom = fraction(phase_solution[i])
        print(f"d2{i+1}_numerator =", numer)
        print(f"d2{i+1}_denominator =", denom)
    numer, denom = fraction(simplify(predictions_sym[3] / -6))
    print(f"Walpha_numerator =", numer)
    print(f"Walpha_denominator =", denom)

    print ("# numerical values for debugging purposes:")
    for i in range(6):
        numer, denom = fraction(phase_solution[i])
        print(f"true_d2{i+1}_numerator =", sym_to_num(numer))
        print(f"true_d2{i+1}_denominator =", sym_to_num(denom))
    numer, denom = fraction(simplify(predictions_sym[3] / -6))
    print(f"true_Walpha_numerator =", sym_to_num(numer))
    print(f"true_Walpha_denominator =", sym_to_num(denom))


dump_solutions_in_python() ; exit()

# p1i in the code, d_{2 i+1} in the paper, sorry.
for i in range(6):
    dump_eq("d_{2"+str(i+1)+"}", phase_solution[i])

# A in the code, W in the paper, sorry.
dump_eq("W", predictions_sym[3])
