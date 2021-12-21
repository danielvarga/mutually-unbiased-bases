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


A = 0.42593834
B = 0.35506058
C = 0.38501704
D = 0.40824829
numerical_magic_constants = [36 * c **2 for c in [A, B, C]]

def reconstruct_tripartition(b0, b1):
    magic_constants = [1, 2, 3]
    msquared = np.abs(np.conjugate(b0.T) @ b1) ** 2
    masks = [np.isclose(msquared, numerical_magic_constants[i]).astype(int) for i in range(3)]
    assert np.all(masks[0] | masks[1] | masks[2])
    magic_matrix = sum(magic_constants[i] * masks[i] for i in range(3))
    return magic_matrix, magic_constants

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


np.set_printoptions(precision=5, suppress=True, linewidth=100000)
for (i, j) in [(0, 1), (1, 2), (2, 0)]:
    prod = np.conjugate(mub[i].T) @ mub[j]
    # deconstruct_product(prod)
    print(f"| B_{i}^dagger B_{j} | / 6")
    print(np.abs(prod) / 6)
    print(angler(prod))


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
from sympy import symbols, Matrix, Transpose, conjugate, expand, simplify, sqrt
from sympy.physics.quantum.dagger import Dagger


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

prod01 = Dagger(symbolic_bases[0]) @ symbolic_bases[1]
prod01 = simplify_roots(prod01)
for i in range(6):
    for j in range(6):
        assert prod01[i, j] - prod01[(i + 2) % 6, (j + 2) % 6] == 0


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
print("eq6prime numerically verified:", (d_1 * magic_vector6).sum())


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
constraints = np.stack([magic_vector1, magic_vector2, magic_vector3, magic_vector5, magic_vector6])

print(constraints)
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
