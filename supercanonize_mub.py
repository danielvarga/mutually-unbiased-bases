# input: three lines of canonized_mubs, like
# cat canonized_mubs | grep mub_100.npy | python analyze_canonized_mubs.py

import sys
import numpy as np
import matplotlib.pyplot as plt

# filename normalized/mub_100.npy i 1 D_l [  -0.      -26.7955 -173.147   -38.4863 -168.1707 -150.0384] P_l [0 5 2 3 1 4] x 55.11879687165239 y 0.0005514574849221608 P_r [4 1 2 3 0 5] D_r [0. 0. 0. 0. 0. 0.] distance 4.003481295679986e-06


TP = 2 * np.pi
TIP = 2j * np.pi
W = np.exp(TIP / 3)


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


def angler(x):
    return np.angle(x) * 180 / np.pi


def deconstruct_product(prod):
    sub0 = prod[:2, :2].copy()
    sub0 /= np.abs(sub0)
    sub1 = prod[:2, 2:4].copy()
    sub1 /= np.abs(sub1)
    sub2 = prod[:2, 4:].copy()
    sub2 /= np.abs(sub2)
    print(np.around(angler(sub1 / sub0)))
    print(np.around(angler(sub2 / sub0)))


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

loss = loss_function(mub)
print("loss", loss)


import sympy
from sympy import symbols, Matrix, Transpose, conjugate, expand, simplify, sqrt, Rational
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
def create_basis(indx, factors):
    left_phases, left_pm, x, y, right_pm, right_phases = factors
    # we count from 0, but the 0th is always 1.
    left_phases_var = [1] + [sympy.symbols(f'p{indx}{i}') for i in range(1, 6)]
    x_var = sympy.symbols('x')
    # TODO assert that x and y behaves acccordingly
    if indx == 0:
        x_here, y_here = x_var, x_var
    elif indx == 1:
        x_here, y_here = x_var, 1
    else:
        assert indx == 2
        x_here, y_here = -x_var, 1
    return left_phases_var, symbolic_generalized_fourier_basis(left_phases_var, left_pm, x_here, y_here)


symbolic_bases = []
for indx in range(3):
    factors = factorized_mub[indx]
    left_phases_var, b = create_basis(indx, factors)
    if indx == 0:
        for v in left_phases_var:
            b = b.subs(v, 1)
    symbolic_bases.append(b)


# all the below is true for normalized/mub_10040.npy and not much else.
# the goal is to reverse engineer this single one. the rest is combinatorics,
# the manifold is zero dimensional.

prod01sym = Dagger(symbolic_bases[0]) @ symbolic_bases[1]
prod01sym = simplify_roots(prod01sym)


A = 0.42593834
B = 0.35506058
C = 0.38501704
D = 0.40824829
numerical_magic_constants = [36 * c **2 for c in [A, B, C]]



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
# other than that, we a general, so it would not be hard to cover all qmubs.
def reconstruct_product(b0, b1):
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
    masks = [np.isclose(normed, all_candidates[i, j], atol=1e-5).astype(int) for i in range(3) for j in range(12)]
    masks = np.array(masks).reshape((3, 12, 6, 6))
    assert masks.sum() == 36, "some elements of the product could not be indentified"
    WWsym = sympy.symbols('WW') # 12th root of unity.
    # TODO we can do it without increasing the degree
    roots_sym = Matrix([WWsym ** i for i in range(12)])
    alphasym, deltasym = sympy.symbols('alpha delta')
    candidates_sym = [alphasym, alphasym * deltasym, alphasym * conjugate(deltasym)]

    normed_prod_sym = sympy.ones(6, 6)
    for i in range(6):
        for j in range(6):
            onehot = masks[:, :, i, j]
            element_index = np.argmax(onehot.sum(axis=1))
            rotation = np.argmax(onehot.sum(axis=0))
            assert onehot[element_index, rotation] == 1
            normed_prod_sym[i, j] *= candidates_sym[element_index] * roots_sym[rotation]
    print(normed_prod_sym)

    msquared = np.abs(prod) ** 2
    masks = [np.isclose(msquared, numerical_magic_constants[i]).astype(int) for i in range(3)]
    masks = np.array(masks)
    magnitudes_sym = sympy.symbols('A B C')
    magnitude_matrix_sym = sympy.ones(6, 6)
    for i in range(6):
        for j in range(6):
            element_index = np.argmax(masks[:, i, j])
            magnitude_matrix_sym[i, j] *= magnitudes_sym[element_index]
    print(magnitude_matrix_sym)
    return matrix_multiply_elementwise(normed_prod_sym, magnitude_matrix_sym)



prod_sym = reconstruct_product(mub[0], mub[1])
print(prod_sym)
exit()


# this is not true at all for the new triple-F(a,0) bases:
'''
for i in range(6):
    for j in range(6):
        assert prod01[i, j] - prod01[(i + 2) % 6, (j + 2) % 6] == 0
'''



# this is the identity that prod01[0, 1] * -W^2 == prod01[0, 3]:
b_rot = - Wsym ** 2 * prod01[0, 1]
d = prod01[0, 3]
print("eq1:", simplify_roots(expand(d - b_rot)))
# -> W**2*p13 - W**2 + W*p11*x - W*p14*x + p11*x + 2*p12 + p13 - p14*x - 2*p15 - 1 = 0

d_1 = factorized_mub[1][0]
x = factorized_mub[0][2]
magic_vector1 = np.array([-W, W**2 * x, -2, W, -W**2 * x, 2])
print("eq1 numerically verified:", (d_1 * magic_vector1).sum())


# this is the identity that prod01[0, 0] * W == prod[0, 4]:
a_rot = Wsym * prod01[0, 0]
e = prod01[0, 4]
print("eq2:", simplify_roots(expand(e - a_rot)))
# -> W**2*p11 + W**2*p14 - W*p11 - W*p13 - W*p14 - W + p13 + 1

magic_vector2 = np.array([1 - W, W**2 - W, 0, 1 - W, W**2 - W, 0])
print("eq2 numerically verified:", (d_1 * magic_vector2).sum())


# this is the identity that prod01[1, 2] * -W^2 == prod01[1, 4]:
rot = - Wsym ** 2 * prod01[1, 2]
target = prod01[1, 4]
print("eq3:", simplify_roots(expand(target - rot)))
# -> -W**2*p13 + W**2 - W*p11*conjugate(x) + W*p14*conjugate(x) - p11*conjugate(x) + 2*p12*conjugate(x) - p13 + p14*conjugate(x) - 2*p15*conjugate(x) + 1
# we multiply by x to get rid of the conjugates:
magic_vector3 = np.array([(W ** 2 + 1) * x, (- W - 1), 2, (- W ** 2 - 1) * x, (W + 1), -2])
print("eq3 numerically verified:", (d_1 * magic_vector3).sum())


# this is the identity that prod01[1, 1] * W == prod01[1, 5]:
rot = Wsym * prod01[1, 1]
target = prod01[1, 5]
# this is the first time we have to use |x| = 1:
print("eq4:", enforce_norm_one(simplify_roots(expand(target - rot)), [sympy.symbols('x')]))
# -> -W**2*p11 - W**2*p14 + W*p11 + W*p13 + W*p14 + W - p13 - 1
# -> unfortunately it's minus magic_vector2, redundant.
#    this suggests that prod01[1, 4] / prod01[1, 0] conveys the same information as prod01[0, 5] / prod01[0, 1] aka eq6.

magic_vector4 = np.array([W - 1, W - W ** 2, 0, W - 1, - W**2 + W, 0])
print("eq4 numerically verified:", (d_1 * magic_vector4).sum())


# this is the identity that prod01[0, 0] * np.exp(1j * 150 / 180 * np.pi) / BIG  == prod01[0, 2] / SMALL:
# BIG aka A = 0.42593834, SMALL aka B = 0.35506058
print("eq5: (", prod01[0, 0], ") * np.exp(1j * 150/180*pi) / A * B =", prod01[0, 2])
# -> p11 + p12 + p13 + p14 + p15 + 1 * np.exp(1j * 150 / 180 * np.pi) / A * B = W**2*p12 + W**2*p15 + W*p11 + W*p14 + p13 + 1

# print(np.exp(1j * 150/180*np.pi) / A * B)
magic_vector5_right = np.array([1, W, W ** 2, 1, W, W ** 2])
print("eq5 numerically verified:", d_1.sum() * np.exp(1j * 150 / 180 * np.pi) / A * B - (d_1 * magic_vector5_right).sum())

magic_vector5 = np.ones(6) * np.exp(1j * 150 / 180 * np.pi) / A * B - magic_vector5_right
print("eq5prime numerically verified:", (d_1 * magic_vector5).sum())


# this is the identity that prod01[0, 1] * - W / BIG  == prod01[0, 5] / MED:
print("eq6: (", simplify_roots(expand(- Wsym * prod01[0, 1])), ") * C / A =", prod01[0, 5])
# eq6: ( -W**2*p12 + W**2*p15 - W*p13 + W - p11*x + p14*x ) * C / A = W**2*p12 - W**2*p15 + W*p11*x - W*p14*x + p13 - 1

magic_vector6_left  = np.array([ W,   - x, - W ** 2, - W,       x,   W ** 2])
magic_vector6_right = np.array([-1, W * x,   W ** 2,   1, - W * x, - W ** 2])
print("eq6 numerically verified:", (d_1 * magic_vector6_left).sum() * C / A - (d_1 * magic_vector6_right).sum())

magic_vector6 = magic_vector6_left * C / A - magic_vector6_right
print("magic_vector6", magic_vector6)
print("eq6prime numerically verified:", (d_1 * magic_vector6).sum())


# this is extracted from the raw empirical observation of D_1. it starts with 0, -120 degrees.
magic_vector7 = np.array([- 1, W, 0, 0, 0, 0])
print("eq7 numerically verified:", (d_1 * magic_vector7).sum())


# this is extracted from the raw empirical observation of D_1. D_1[3] = 120+D_1[4], where D_1[4] ~ 7.355.
magic_vector8 = np.array([0, 0, 0, -1, W, 0])
print("eq8 numerically verified:", (d_1 * magic_vector8).sum())


# according to the svd rank analysis, eq7 and eq8 are not equivalent to any of the rest
# in isolation, but adding them to eqs eq1-eq6 does not increase the rank (it's still 4).
# a maximum rank span is eq1, eq5, eq7, eq8.
# eq5 is the tricky one with the magnitude ratio, but eqs 1-7-8 already give us
# something nontrivial according to linsolve:
# p12 = W*x/2 - W/2 + p14*(-W**2*x/2 + W**2/2) + p15
# manually that's
# p12 = (1 - x) / 2 * W**2 * (p14 - W**2) + p15
# why is this even a phase? this is our first identity moving beyond the
# form "element of prod12 is constant times another element" (eqs 1-6) or
# from "element of D_1 is constant times another element" (eqs 7-8).

p14, p15 = d_1[4:]
p12supposedly = (1 - x) / 2 * W**2 * (p14 - W**2) + p15

print("equation linsolve'd from eq1-7-8 is numerically verified:", d_1[2] - p12supposedly)



# constraints = np.stack([magic_vector3, magic_vector6])
# eq2 and eq4 exactly sum to zero symbolically.
# eq3 and eq6 seem to be numerically redundant from the perspective of D_1,
# (with empirical x, A, C substituted), but that can actually
# make them valuable from the perspective of getting constraints
# about x, A, C. not sure they are actually redundant, though. it's very imprecise.
# the second three is always minus the first three, so we only consider the first three.
def whats_up_with_eq3_vs_eq6():
    magic_vector3 = np.array([(W ** 2 + 1) * x, (- W - 1), 2])
    magic_vector6_left  = np.array([ W,   - x, - W ** 2])
    magic_vector6_right = np.array([-1, W * x,   W ** 2])
    magic_vector6 = magic_vector6_left * C / A - magic_vector6_right
    print("magic_vector6 / magic_vector3", magic_vector6 / magic_vector3)
    np.set_printoptions(precision=12, suppress=True, linewidth=100000)
    print(np.abs(magic_vector6 / magic_vector3), angler(magic_vector6 / magic_vector3))


print("linear constraints:")
constraints = np.stack([magic_vector1, magic_vector2, magic_vector3, magic_vector5, magic_vector6, magic_vector7, magic_vector8])
constraints = np.stack([magic_vector1, magic_vector5, magic_vector7, magic_vector8])

# print(constraints)
np.set_printoptions(precision=12, suppress=False, linewidth=100000)
u, s, vh = np.linalg.svd(constraints)
print(s)


# TODO do the rest, there are a couple of hopefully nonredundant ones.
# TODO do even the redundant ones, maybe they are nicer looking.


xvar = symbols('x')
left_phases_var = Matrix([[1] + [sympy.symbols(f'p1{i}') for i in range(1, 6)]]) # row vector

magic1sym = Matrix([-Wsym, Wsym**2 * xvar, -2, Wsym, -Wsym**2 * xvar, 2]) # column vector
print(left_phases_var @ magic1sym)

magic2sym = Matrix([1 - Wsym, Wsym**2 - Wsym, 0, 1 - Wsym, Wsym**2 - Wsym, 0])
print(left_phases_var @ magic2sym)

magic3sym = Matrix([(Wsym ** 2 + 1) * xvar, (- Wsym - 1), 2, (- Wsym ** 2 - 1) * xvar, (Wsym + 1), -2])
print(left_phases_var @ magic3sym)


BperAvar = symbols('BperA')
magic5rightsym = Matrix([1, Wsym, Wsym ** 2, 1, Wsym, Wsym ** 2])
magic5sym = Matrix([1] * 6) * (1j/2 - sqrt(3) / 2) * BperAvar - magic5rightsym
print(simplify(left_phases_var @ magic5sym))


CperAvar = symbols('CperA')
magic6leftsym = Matrix([ Wsym,   - xvar, - Wsym ** 2, - Wsym,    xvar,   Wsym ** 2])
magic6rightsym = Matrix([-1, Wsym * xvar,   Wsym ** 2,   1, - Wsym * xvar, - Wsym ** 2])
magic6sym = magic6leftsym * CperAvar - magic6rightsym

# print(">>>", expand(magic6sym.subs(xvar, x).subs(CperAvar, C/A).subs(Wsym, -0.5 - 3 ** 0.5 / 2j)))


magic7sym = Matrix([-1, Wsym, 0, 0, 0, 0])
magic8sym = Matrix([0, 0, 0, -1, Wsym, 0])


def sym_to_num(formula):
    f = formula.subs(Wsym, W)
    f = f.subs(xvar, x).subs(BperAvar, B/A).subs(CperAvar, C/A)
    for i in range(1, 6):
        f = f.subs(left_phases_var[i], d_1[i])
    a = np.array(f, dtype=np.complex128)
    return np.squeeze(a)


np.set_printoptions(precision=12, suppress=True, linewidth=100000)

for i, magicsym in enumerate([magic1sym, magic2sym, magic3sym, None, magic5sym, magic6sym, magic7sym, magic8sym], 1):
    if magicsym is None:
        continue
    print(f"verifying EQ{i}:", np.abs(sym_to_num(left_phases_var @ magicsym)))


from sympy.solvers.solveset import linsolve

p_var = tuple(sympy.symbols(f'p1{i}') for i in range(1, 6))

def apply(magicsym):
    return (left_phases_var @ magicsym)[0]

eq_system = [apply(magicsym) for magicsym in [magic1sym, magic2sym, magic3sym, magic5sym, magic6sym, magic7sym, magic8sym]]
eq_system = [apply(magicsym) for magicsym in [magic1sym, magic2sym, magic5sym, magic7sym, magic8sym]]
eq_system = [apply(magicsym) for magicsym in [magic2sym, magic5sym, magic7sym, magic8sym]]
eq_system = [eq.subs(Wsym, Rational(- 1, 2) - 1j * sqrt(3) * Rational(1, 2)) for eq in eq_system]
print(eq_system)

print("=======")

sols = linsolve(eq_system, p_var)

sols = list(sols)[0]
sols = [expand(v).factor(BperAvar) for v in sols]
for i, v in enumerate(sols):
    print(i + 1, v)
