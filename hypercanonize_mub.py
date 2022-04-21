# input: three lines of canonized_mubs, like
# cat canonized_mubs | grep mub_100.npy | python analyze_canonized_mubs.py

import sys
import numpy as np
import matplotlib.pyplot as plt

import sympy
from sympy import symbols, Matrix, Transpose, conjugate, sqrt, Rational
from sympy import expand, factor, cancel, nsimplify, simplify, fraction
from sympy.physics.quantum.dagger import Dagger
from sympy.matrices.dense import matrix_multiply_elementwise

# filename normalized/mub_100.npy i 1 D_l [  -0.      -26.7955 -173.147   -38.4863 -168.1707 -150.0384] P_l [0 5 2 3 1 4] x 55.11879687165239 y 0.0005514574849221608 P_r [4 1 2 3 0 5] D_r [0. 0. 0. 0. 0. 0.] distance 4.003481295679986e-06


TP = 2 * np.pi
TIP = 2j * np.pi
W = np.exp(TIP / 3)


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


A = 0.42593834 # new name W
B = 0.35506058 # new name U
C = 0.38501704 # new name V
D = 0.40824829
numerical_magic_constants = [36 * c **2 for c in [A, B, C]]


def test_UVW_constraints():
    # name clash with sympy.sqrt:
    from numpy import sqrt
    assert np.isclose(4*A**2+B**2+C**2, 1)
    assert np.isclose(2*A**2, B**2 + sqrt(3)*B*C)
    assert np.isclose(C, 1 - sqrt(3) * B)
    assert np.isclose(A, sqrt((sqrt(3) - 2*B)*B)/sqrt(2))
    assert np.isclose(A**2 + B**2, sqrt(3)/2*B)
    print("passed all test regarding U, V, W constraints")


# test_UVW_constraints() ; exit()


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

    '''
    plt.scatter(collection.real, collection.imag)
    plt.scatter(alpha_circle.real, alpha_circle.imag)
    plt.show()
    plt.scatter(angler(collection), np.zeros_like(collection))
    plt.scatter(angler(alpha_circle), np.zeros_like(alpha_circle))
    plt.show()
    exit()
    '''

    # more exactly delta union conj(delta) circle:
    delta_circle = collection[neighbor_counts == 6]
    assert len(delta_circle) == 12
    # after the previous division by alpha,
    # any element of the delta_circle will do as delta.
    # we break the tie by choosing the one closest to 0,
    # that's supposed to be either ~7.3737 degrees or ~60-7.3737 degrees.
    print(angler(delta_circle))
    dist_to_0 = np.abs(np.angle(delta_circle))
    i = np.argmin(dist_to_0)
    delta = delta_circle[i]
    return alpha, delta


def reconstruct_product_generally(b0, b1, use_alpha):
    alpha, delta = identify_alpha_delta(b0, b1)
    # by the time we get to this, alpha is moved out,
    # and delta is 7.3737 degrees.
    if not use_alpha:
        assert np.isclose(alpha, 1)

    prod = np.conjugate(b0.T) @ b1
    msquared = np.abs(prod) ** 2
    normed = prod.copy()
    normed /= np.abs(normed)

    WW = np.exp(TIP / 12)
    roots = WW ** np.arange(12)
    candidates = np.array([alpha, alpha * delta, alpha / delta])
    all_candidates = candidates[:, None] * roots[None, :]
    masks = [np.isclose(normed, all_candidates[i, j], atol=1e-4).astype(int) for i in range(3) for j in range(12)]
    masks = np.array(masks).reshape((3, 12, 6, 6))
    assert masks.sum() == 36, "some elements of the product could not be indentified"
    # TODO we can do it without increasing the degree
    roots_sym = Matrix([WWsym ** i for i in range(12)])
    if use_alpha:
        candidates_sym = [alphasym, alphasym * deltasym, alphasym * conjugate(deltasym)]
    else:
        candidates_sym = [1, deltasym, conjugate(deltasym)]

    normed_prod_sym = sympy.ones(6, 6)
    for i in range(6):
        for j in range(6):
            onehot = masks[:, :, i, j]
            element_index = np.argmax(onehot.sum(axis=1))
            rotation = np.argmax(onehot.sum(axis=0))
            assert onehot[element_index, rotation] == 1
            normed_prod_sym[i, j] *= candidates_sym[element_index] * roots_sym[rotation]
    assert np.allclose(sym_to_num(normed_prod_sym), normed, atol=1e-4)

    # TODO get rid of lots of copypaste
    magnitude_masks = [np.isclose(msquared, numerical_magic_constants[i], atol=1e-2).astype(int) for i in range(3)]
    magnitude_masks = np.array(magnitude_masks)
    assert magnitude_masks.sum() == 36
    magnitude_matrix_sym = sympy.ones(6, 6)
    for i in range(6):
        for j in range(6):
            element_index = np.argmax(magnitude_masks[:, i, j])
            magnitude_matrix_sym[i, j] *= magnitudes_sym[element_index]
    assert np.allclose(sym_to_num(6 * magnitude_matrix_sym), np.abs(prod))
    return matrix_multiply_elementwise(normed_prod_sym, magnitude_matrix_sym)


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


def subs_roots(f):
    f = f.subs(Wsym, - Rational(1, 2) + 1j * sqrt(3) * Rational(1, 2))
    f = f.subs(WWsym, Rational(1, 2) * 1j + sqrt(3) * Rational(1, 2))
    f = expand(f)
    return f


def sym_to_num(formula):
    f = formula.subs(Wsym, W).subs(WWsym, np.exp(TIP/12))
    f = f.subs(magnitudes_sym[0], A).subs(magnitudes_sym[1], B).subs(magnitudes_sym[2], C)
    f = f.subs(deltasym, delta).subs(alphasym, alpha)
    try:
        a = np.array(f, dtype=np.complex128)
    except:
        print("failed to fully evaluate", formula)
        print("still variables left in", f)
        raise
    return np.squeeze(a)


alpha1, delta1 = identify_alpha_delta(mub[0], mub[1])
print("delta_orig", angler(delta1))


alpha_removal = True
if alpha_removal:
    mub[1] /= alpha1
    alpha2, delta2 = identify_alpha_delta(mub[0], mub[2])
    mub[2] /= alpha2

    alpha1, delta1 = identify_alpha_delta(mub[0], mub[1])
    assert np.isclose(alpha1, 1)
    alpha2, delta2 = identify_alpha_delta(mub[0], mub[2])
    assert np.isclose(alpha2, 1)

    alpha0, delta0 = identify_alpha_delta(mub[1], mub[2])
    assert np.isclose(alpha0, 1)


delta = delta1
print("delta", angler(delta))

# dump_products(mub)



'''
for (i, j) in [(0, 1), (1, 2), (2, 0)]:
        prods[i] = np.conjugate(mub[i].T) @ mub[j] / 6
        print(f"X = B_{i}^dagger B_{j} / 6")
        print(prods[i])
        print(np.angle(prods[i]) / 2/ np.pi * 360)
        print(f"X**2 ")
        print(prods[i] @ prods[i])
exit()
'''

prod01 = np.conjugate(mub[0].T) @ mub[1]
prod12 = np.conjugate(mub[1].T) @ mub[2]
prod20 = np.conjugate(mub[2].T) @ mub[0]

np.set_printoptions(precision=3, suppress=True, linewidth=100000)

print("01^2", prod01 @ prod01) # (120deg)Id
print("20^2", prod20 @ prod20) # Id
print("12^2", prod12 @ prod12) # uglier
print("12^3", prod12 @ prod12 @ prod12)


def dump_eigenvalues():
    print("Okay")
    np.set_printoptions(precision=3, suppress=True, linewidth=100000)
    for (i, j) in [(0, 1), (1, 2), (2, 0)]:
        prod = np.conjugate(mub[i].T) @ mub[j]
        eigvals, eigvecs = np.linalg.eig(prod / 6)
        print("filename", filename, "eigvals", i, j, np.sort(angler(eigvals)))


dump_eigenvalues() ; exit()


def rot_to_60(a):
    b = []
    for p in a:
        while not(0 <= np.angle(p) < np.pi / 3):
            p *= np.exp(1j * np.pi / 3)
        b.append(p)
    b = np.array(b)
    return b


for alpha in (-W**2) ** np.arange(6):
    mub1prime = mub[1].copy()
    mub1prime /= alpha
    prod01 = np.conjugate(mub[0].T) @ mub1prime
    # plt.scatter(prod01.flatten().real, prod01.flatten().imag)
    # plt.show()


Wsym, WWsym, alphasym, deltasym = sympy.symbols('W WW alpha delta')
magnitudes_sym = sympy.symbols('A B C')


sym_prods = []
for (i, j) in [(0, 1), (1, 2), (2, 0)]:
    prod = np.conjugate(mub[i].T) @ mub[j] / 6
    vals = prod.flatten()
    # vals = rot_to_60(vals)
    # print(sorted(set(np.around(angler(vals), 3))))
    # plt.scatter(vals.real, vals.imag)
    # plt.show()

    print(i, j)
    alpha, delta = identify_alpha_delta(mub[i], mub[j])
    alpha = 1
    sym_prods.append(reconstruct_product_generally(mub[i], mub[j], use_alpha=False))


for indx, (i, j) in enumerate([(0, 1), (1, 2), (2, 0)]):
    prod = sym_prods[indx]
    if (i, j) == (1, 2):
        exponent = 3
    else:
        exponent = 2
    print("======")
    print("pair", (i, j), "exponent", exponent)
    pp = prod ** exponent
    pp = subs_roots(enforce_norm_one(pp, [deltasym]))
    pp = factor(simplify(pp))
    print(nsimplify(enforce_norm_one(pp, [deltasym])))
    '''
    if exponent == 3:
        print(sympy.latex(nsimplify(pp[0, 0] / 6)))
    '''

print("!!!!!!!", 6*np.sqrt(3)*A**2*B + 6*A**2*C - 3*B**2*C + C**3)

# 0 = 2(-60deg)A^2 + (120deg)B^2 + sqrt(3)(120deg)BC
print("????", -2*W*A**2 + W*B**2 + np.sqrt(3)*W*BC)


exit()


'''
plt.scatter(prod01.flatten().real, prod01.flatten().imag)
plt.show()
plt.scatter(prod12.flatten().real, prod12.flatten().imag)
plt.show()
plt.scatter(prod20.flatten().real, prod20.flatten().imag)
plt.show()
exit()
'''




# dump_products(mub)


# put back B_0=Id in front of them, and also scale them back to be actually unitary.
def put_back_id(mub):
    return np.concatenate([np.eye(6, dtype=np.complex128)[np.newaxis, ...], mub / 6 ** 0.5])


loss = loss_function(mub)
print("loss", loss)

np.save('tmp.npy', put_back_id(mub))
print("normalized qMUB saved into tmp.npy")



# symbol of third root of unity.
Wsym = symbols('W')




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
    assert np.isclose(np.angle(x) / TP * 360, 55.118, atol=1e-3)
    assert np.isclose(y, 1)
    left_phases_var = [sympy.symbols(f'p{indx}{i}') for i in range(6)]
    x_var = sympy.symbols('x')
    return left_phases_var, symbolic_generalized_fourier_basis(left_phases_var, left_pm, x_var, 1)


WWsym = sympy.symbols('WW') # 12th root of unity.
magnitudes_sym = sympy.symbols('A B C')
xvar = symbols('x')
alphasym, deltasym = sympy.symbols('alpha delta')
left_phases_var = Matrix([[sympy.symbols(f'p1{i}') for i in range(6)]]) # row vector

d_1 = factorized_mub[1][0]
x = factorized_mub[0][2]

# at this point we hardwire mub_10024.py, sorry.
# its distinct characteristic is that the tripartition
# looks like this:
# .U..V.
# U..V..
# ..V..U
# .V..U.
# V..U..
# ..U..V
# (U=0.35, V=0.38, .=0.40)
# thanks to this, the top left corner of its normalized form obeys these rules:
# ab.... (alpha, beta, gamma really)
# c.....
# ......
# beta/alpha = alpha/gamma
# delta := beta/alpha
# other than that, we are general, so it would not be hard to cover all qmubs.
def reconstruct_product(b0, b1):
    # ouch. this is right now the simplest way to
    # make sym_to_num be aware of these variables.
    # TODO figure out something more robust.
    global alpha, delta

    prod = np.conjugate(b0.T) @ b1
    msquared = np.abs(prod) ** 2
    normed = prod.copy()
    normed /= np.abs(normed)
    WW = np.exp(TIP / 12)
    roots = WW ** np.arange(12)
    alpha = normed[0, 0]
    beta = normed[0, 1]
    delta = beta / alpha
    assert np.isclose(normed[1, 0], alpha / delta), "this is a restriction currently, please use mub_10024.py or generalize"
    candidates = np.array([alpha, alpha * delta, alpha / delta])
    all_candidates = candidates[:, None] * roots[None, :]
    masks = [np.isclose(normed, all_candidates[i, j], atol=1e-4).astype(int) for i in range(3) for j in range(12)]
    masks = np.array(masks).reshape((3, 12, 6, 6))
    assert masks.sum() == 36, "some elements of the product could not be indentified"
    # TODO we can do it without increasing the degree
    roots_sym = Matrix([WWsym ** i for i in range(12)])
    candidates_sym = [alphasym, alphasym * deltasym, alphasym * conjugate(deltasym)]

    normed_prod_sym = sympy.ones(6, 6)
    for i in range(6):
        for j in range(6):
            onehot = masks[:, :, i, j]
            element_index = np.argmax(onehot.sum(axis=1))
            rotation = np.argmax(onehot.sum(axis=0))
            assert onehot[element_index, rotation] == 1
            normed_prod_sym[i, j] *= candidates_sym[element_index] * roots_sym[rotation]
    assert np.allclose(sym_to_num(normed_prod_sym), normed)

    msquared = np.abs(prod) ** 2
    masks = [np.isclose(msquared, numerical_magic_constants[i], atol=1e-4).astype(int) for i in range(3)]
    masks = np.array(masks)
    assert masks.sum() == 36
    magnitude_matrix_sym = 6 * sympy.ones(6, 6)
    for i in range(6):
        for j in range(6):
            element_index = np.argmax(masks[:, i, j])
            magnitude_matrix_sym[i, j] *= magnitudes_sym[element_index]
    assert np.allclose(sym_to_num(magnitude_matrix_sym), np.abs(prod))
    return matrix_multiply_elementwise(normed_prod_sym, magnitude_matrix_sym)




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

alpha = 0.8078018037463457+0.589454193185654j
delta = -0.12834049204788406+0.9917301639563593j

symbolic_bases = create_symbolic_mub(factorized_mub)

assert np.allclose(sym_to_num(symbolic_bases[1]), mub[1])


# alpha and delta returned in global variables
# because of the sym_to_num closure, sorry.
prod01reconstructed_sym = reconstruct_product_generally(mub[0], mub[1])
assert np.allclose(sym_to_num(prod01reconstructed_sym), np.conjugate(mub[0].T) @ mub[1])


# alpha now recieved from reconstruct_product_generally() as global.
# we remove it from mub[1], and correct the left phase accordingly.

# d_1 /= alpha
mub[1] = factors_to_basis(factorized_mub[1])

# now we have mutated mub[1], we have to do it all over again
prod01reconstructed_sym = reconstruct_product_generally(mub[0], mub[1])

# assert np.isclose(alpha, 1) # second time's the charm
assert np.allclose(sym_to_num(prod01reconstructed_sym), np.conjugate(mub[0].T) @ mub[1])


symbolic_bases = create_symbolic_mub(factorized_mub)

# why range(2) not range(3)?
# we don't do anything with B_2 (the last basis) now,
# so we don't even substitute its phases.
# TODO we should.
for i in range(2):
    assert np.allclose(sym_to_num(symbolic_bases[i]), mub[i])

# this is where we collect the D_1 elements written up in terms of A,B,C,x,W,WW.
phase_solution = left_phases_var[:]

print('''HOPPA HOPPA EZEK IMMARON NEM IGAZAK, HA KIFORGATTAM alpha-t BALRA!!!
csak ph[0] es ph[1] hanyadosarol tudok nyilatkozni, ugyanugy, mint ahogy a ph[3] es ph[4] viszonyarol eddig is.''')

symbolic_bases[1] = symbolic_bases[1].subs(left_phases_var[0], conjugate(xvar))
phase_solution[0] = conjugate(xvar)
assert np.allclose(sym_to_num(symbolic_bases[1]), mub[1], atol=1e-4)
symbolic_bases[1] = symbolic_bases[1].subs(left_phases_var[1], -Wsym ** 2)
phase_solution[1] = -Wsym ** 2
assert np.allclose(sym_to_num(symbolic_bases[1]), mub[1])
symbolic_bases[1] = symbolic_bases[1].subs(left_phases_var[3], -Wsym * conjugate(xvar) * left_phases_var[4])
phase_solution[3] = -Wsym * conjugate(xvar) * left_phases_var[4]
assert np.allclose(sym_to_num(symbolic_bases[1]), mub[1])
symbolic_bases[1] = simplify_roots(enforce_norm_one(symbolic_bases[1], [xvar]))

print("B_1", symbolic_bases[1])


prod01sym = Dagger(symbolic_bases[0]) @ symbolic_bases[1]
prod01sym = simplify_roots(prod01sym)
print("product", prod01sym)

assert np.allclose(sym_to_num(prod01sym), np.conjugate(mub[0].T) @ mub[1])

diff = prod01sym - prod01reconstructed_sym

assert np.allclose(sym_to_num(diff), 0, atol=1e-4)


diff = subs_roots(diff)

assert np.allclose(sym_to_num(diff), 0, atol=1e-4)


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


# variables = [sympy.symbols(f'p1{i}') for i in range(6)]
# the remaining variables after the substitutions above:
variables = sympy.symbols('p12 p14 p15 A B C')

all_directions = []
all_biases = []
for i in range(6):
    for j in range(6):
        direction, bias = extract_directions(diff[i, j], variables)
        all_directions.append(direction)
        all_biases.append(bias)

all_directions = Matrix([[coeff[0] for coeff in direction.tolist()] for direction in all_directions])
all_biases = Matrix(all_biases)


import matplotlib.pyplot as plt


def show_points(points, title):
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


bases = [ #  BFE
    (0, 0), # nicest of the 6 As, tells about A alpha
    (3, 3), # nicest of the 6 Cs, tells about A alpha
    (2, 2), # nicest of the 6 Es, tells about C alpha
    (0, 1), # nicest of the 3 Bs, tells about B alpha delta
    (1, 0), # nicest of the 3 Fs, tells about B alpha conj(delta)
    (2, 3), # nicest of the 2x3 Ds, tells about A alpha delta
    (3, 2), # nicest of the 2x3 Gs, tells about A alpha conj(delta)
]

directions = []
biases = []
# note: if we keep the first row and the first element of the second row.
# that's already rank 6, enough to determine all our variables.
for (i, j) in bases:
        direction, bias = extract_directions(diff[i, j], variables)
        directions.append(direction)
        biases.append(bias)

# a python list of column vector sympy Matrices is not trivial to convert to a sympy Matrix:
directions = Matrix([[coeff[0] for coeff in direction.tolist()] for direction in directions])
biases = Matrix(biases)

# show_directions(directions, biases) ; exit()

directions[5, :] /= - 1j ; biases[5] /= - 1j
directions[6, :] /= 1j   ; biases[6] /= 1j

directions = directions.subs(alphasym, nsimplify(-1) / 6)
directions = expand(directions)
biases = expand(biases)

print("---")

def dump_eq(name, eq):
    print("\\begin{equation}")
    print(name, "=", sympy.latex(nsimplify(eq)))
    print("\\end{equation}")
    print()


dump_eq("A", directions)
dump_eq("x", variables)
dump_eq("b", biases)

print("getting rid of the redundant directions[4], and postponing solving for U and V")
directions = directions.col_del(5)
directions = directions.col_del(4)

directions = directions.row_del(4)
directions = directions.row_del(3)
directions = directions.row_del(2)
biases = biases.row_del(4)
biases = biases.row_del(3)
biases = biases.row_del(2)
variables = variables[:4]

dump_eq("A", directions)
dump_eq("x", variables)
dump_eq("b", biases)


directions_num = sym_to_num(directions)
biases_num = sym_to_num(biases)

np.set_printoptions(precision=12, suppress=False, linewidth=100000)
u, s, vh = np.linalg.svd(directions_num)
print("singular values:", s)

predictions = np.linalg.lstsq(directions_num, -biases_num, rcond=None)[0]
# expectations = np.array(d_1[[2, 4, 5]].tolist() + [A, B, C]) # this was before eliminating B, C them.
expectations = np.array(d_1[[2, 4, 5]].tolist() + [A * alpha * (- 6)])
assert np.allclose(predictions, expectations, atol=1e-4)
print("passed the test: D_1, A, B, C can be reconstructed from these elements of the product")

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

for i in range(4):
    dump_eq(str(variables[i]), predictions_sym[i])


phase_solution[2] = predictions_sym[0]
phase_solution[4] = predictions_sym[1]
phase_solution[5] = predictions_sym[2]
phase_solution[1] = subs_roots(phase_solution[1])
# phase_solution[3] still has some p14 in it, let's get rid of it. also W:
phase_solution[3] = subs_roots(expand(phase_solution[3].subs(left_phases_var[4], phase_solution[4])))

# turning a/c + b/c into (a+b)/c, much much shorter:
for i in range(6):
    phase_solution[i] = sympy.polys.rationaltools.together(phase_solution[i])
    phase_solution[i] = sympy.polys.polytools.factor(phase_solution[i], gaussian=True)
    # phase_solution[i] = sympy.polys.polytools.cancel(phase_solution[i])

phase_solution = Matrix(phase_solution)

assert np.allclose(sym_to_num(phase_solution), d_1, atol=1e-4)

print("congratulations! D_1 (hence B_1) and A completely written up in terms of alpha, delta, x. B, C could be written up, if needed.")


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
