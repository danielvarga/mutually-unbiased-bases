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


# later we will probably generalize
assert len(mubs.keys()) == 1
factorized_mub = mubs[filename]
assert len(factorized_mub) == 3



import sympy
from sympy import symbols, Matrix, Transpose, conjugate
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


def test_symbolic_fourier_basis():
    x_var, y_var = sympy.symbols('x y')
    b = symbolic_fourier_basis(x_var, y_var)
    prod = Dagger(b) @ b
    prod = simplify_roots(prod)
    prod = enforce_norm_one(prod, [x_var, y_var])
    prod = apply_elemwise(lambda expr: sympy.factor(expr).subs(1 + Wsym + Wsym ** 2, 0), prod)
    assert prod == 6 * sympy.eye(6)


# test_symbolic_fourier_basis() ; exit()


# we don't do right phase, it is identity in the cases we care about.
def symbolic_generalized_fourier_basis(left_phases_var, left_pm, x_var, y_var, right_pm):
    fourier = symbolic_fourier_basis(x_var, y_var)
    return sympy.diag(left_phases_var, unpack=True) @ left_pm.astype(int) @ fourier @ right_pm


# indx is needed to create the variables
def create_basis(indx, factors):
    left_phases, left_pm, x, y, right_pm, right_phases = factors
    # we count from 0, but the 0th is always 1.
    left_phases_var = [1] + [sympy.symbols(f'p{indx}{i}') for i in range(1, 6)]
    x_var, y_var = sympy.symbols(f'x{indx} y{indx}')
    return left_phases_var, [x_var, y_var], symbolic_generalized_fourier_basis(left_phases_var, left_pm, x_var, y_var, right_pm)


A = 0.42593834
B = 0.35506058
C = 0.38501704
D = 0.40824829
numerical_magic_constants = [36 * c **2 for c in [A, B, C]]

# asquare = (6*A) ** 2 etc.
def reconstruct_tripartition(b0, b1):
    magic_constants = sympy.symbols('asquare bsquare csquare')
    msquared = np.abs(np.conjugate(b0.T) @ b1) ** 2
    masks = [np.isclose(msquared, numerical_magic_constants[i]).astype(int) for i in range(3)]
    assert np.all(masks[0] | masks[1] | masks[2])
    magic_matrix = sum(magic_constants[i] * masks[i] for i in range(3))
    return magic_matrix, magic_constants


# creates a symbolic matrix. this being equal to 0 is equivalent to
# |B_0^dagger B_1 / 6| ^ 2 being that particular weird matrix with distinct repeating elements A^2 B^2 C^2.
# variables of the equation system are p[012][12345], x0, asquared, bsquared, csquared.
# additionally, we have a set of simple quadratic equations:
# p[012][12345] and x0 are phases, that is, e.g., x0*conj(x0)==1,
# and asquared, bsquared, csquared are positive real numbers.
def write_up_equation_for_single_product(factorized_mub):
    b0 = factors_to_basis(factorized_mub[0])
    b1 = factors_to_basis(factorized_mub[1])
    magic_matrix_01, magic_constants = reconstruct_tripartition(b0, b1)

    left_phases0_var, xy0_var, b0 = create_basis(0, factorized_mub[0])
    left_phases1_var, xy1_var, b1 = create_basis(1, factorized_mub[1])

    # x0=-x1, y0=y1=0
    b0 = b0.subs(xy0_var[1], 1).subs(xy1_var[1], 1).subs(xy1_var[0], - xy0_var[0])
    b1 = b1.subs(xy0_var[1], 1).subs(xy1_var[1], 1).subs(xy1_var[0], - xy0_var[0])

    prod = Dagger(b0) * b1
    prod = simplify_roots(prod)
    prod = enforce_norm_one(prod, left_phases0_var + xy0_var + left_phases1_var + xy1_var)

    prod = sympy.expand(Dagger(prod).multiply_elementwise(prod))
    prod = simplify_roots(prod)
    prod = enforce_norm_one(prod, left_phases0_var + xy0_var + left_phases1_var + xy1_var)
    prod = sympy.expand(prod)
    prod = apply_elemwise(lambda expr: sympy.factor(expr).subs(1 + Wsym + Wsym ** 2, 0), prod)
    prod = sympy.expand(prod)
    prod -= magic_matrix_01
    print(prod)
    return prod


# prod = write_up_equation_for_single_product(factorized_mub); exit()


# creates 3 symbolic matrices. these being all equal to 0 is equivalent to the set being quasi-MUB.
# |B_i^dagger B_j / 6| ^ 2 being those particular weird matrices with distinct repeating elements A^2 B^2 C^2.
# variables of the equation system are p[012][12345], x0, asquared, bsquared, csquared.
# additionally, we have a set of simple quadratic equations:
# p[012][12345] and x0 are phases, that is, e.g., x0*conj(x0)==1,
# and asquared, bsquared, csquared are positive real numbers.
def write_up_equation_for_all_products(factorized_mub):
    b0_numeric = factors_to_basis(factorized_mub[0])
    b1_numeric = factors_to_basis(factorized_mub[1])
    b2_numeric = factors_to_basis(factorized_mub[2])
    magic_matrix_01, magic_constants = reconstruct_tripartition(b0_numeric, b1_numeric)
    magic_matrix_12, magic_constants = reconstruct_tripartition(b1_numeric, b2_numeric)
    magic_matrix_20, magic_constants = reconstruct_tripartition(b2_numeric, b0_numeric)

    basis_list = []
    left_phases_var_list = []
    xy_vars_list = []
    for i in range(3):
        left_phases_var, xy_var, basis = create_basis(i, factorized_mub[i])
        left_phases_var_list.append(left_phases_var)
        xy_vars_list.append(xy_var)
        basis_list.append(basis)

    b0, b1, b2 = basis_list

    x0 = xy_vars_list[0][0]
    b0 = b0.subs(xy_vars_list[0][1], 1) # y0=1
    b1 = b1.subs(xy_vars_list[1][1], 1).subs(xy_vars_list[1][0], - x0) # y1=1, x1=-x0
    b2 = b2.subs(xy_vars_list[2][0], x0).subs(xy_vars_list[2][1], x0) # x2=x0, y2=x0

    def magnitudes(prod):
        prod = simplify_roots(prod)
        left_phases_vars = [left_phases_var for left_phases_vector in left_phases_var_list for left_phases_var in left_phases_vector if left_phases_var != 1]
        prod = enforce_norm_one(prod, left_phases_vars + [x0])

        prod = sympy.expand(Dagger(prod).multiply_elementwise(prod))
        prod = simplify_roots(prod)
        prod = enforce_norm_one(prod, left_phases_vars + [x0])
        prod = sympy.expand(prod)
        # that's the slow part, it's also perfectly useless here.
        # prod = apply_elemwise(lambda expr: sympy.factor(expr).subs(1 + Wsym + Wsym ** 2, 0), prod)
        prod = sympy.expand(prod)
        return prod

    prod01 = Dagger(b0) * b1
    magnitudes01 = magnitudes(prod01)
    magnitudes01 -= magic_matrix_01
    print("done matrix 01", file=sys.stderr)

    prod12 = Dagger(b1) * b2
    magnitudes12 = magnitudes(prod12)
    magnitudes12 -= magic_matrix_12
    print("done matrix 12", file=sys.stderr)

    prod20 = Dagger(b2) * b0
    magnitudes20 = magnitudes(prod20)
    magnitudes20 -= magic_matrix_20
    print("done matrix 20", file=sys.stderr)

    print(magnitudes01)
    print(magnitudes12)
    print(magnitudes20)


prod = write_up_equation_for_all_products(factorized_mub) ; exit()


factors = factorized_mub[0]
left_phases, left_pm, x, y, right_pm, right_phases = factors

left_phases_var = sympy.symbols('p0 p1 p2 p3 p4 p5')
x_var, y_var = sympy.symbols('x y')
fb = symbolic_generalized_fourier_basis(left_phases_var, left_pm, x_var, y_var, right_pm)
print(fb)
exit()







mub = []
matrices = [] # rows are tuples of 
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


for (i, j) in [(0, 1), (1, 2), (2, 0)]:
    prod = np.conjugate(mub[i].T) @ mub[j]
    # print(f"| B_{i}^dagger B_{j} | / 6")
    # print(np.abs(prod) / 6)


loss = loss_function(mub)
print("loss", loss)


# normalized/mub_120.npy fourier parameters:
# a, -b
# -a-b, -b
# a, a+b
# where a = 55.11476631781354, b = 0.007509388499017998
# let's add something to both, and see what happens:
def perturb_fourier_parameters(factorized_mub, alpha_perturbation_degree, beta_perturbation_degree):
    perturbed_mub = []
    alpha = np.exp(1j * np.pi / 180 * alpha_perturbation_degree)
    beta = np.exp(1j * np.pi / 180 * beta_perturbation_degree)

    for i in range(3):
        factors = factorized_mub[i]
        left_phases, left_pm, x, y, right_pm, right_phases = factors
        perturbed_x = x ; perturbed_y = y
        if i == 0:
            perturbed_x *= alpha
        elif i == 1:
            perturbed_x /= alpha
        elif i == 2:
            perturbed_x *= alpha
            perturbed_y *= alpha
        if i == 0:
            perturbed_y /= beta
        elif i == 1:
            perturbed_x /= beta
            perturbed_y /= beta
        elif i == 2:
            perturbed_y *= beta
        basis = factors_to_basis((left_phases, left_pm, perturbed_x, perturbed_y, right_pm, right_phases))
        perturbed_mub.append(basis)
    perturbed_mub = np.array(perturbed_mub)
    return perturbed_mub


alpha_perturbation_degrees = np.linspace(-0.01, 0.01, 100)
beta_perturbation_degrees = np.linspace(-0.01, 0.01, 100)
X, Y = np.meshgrid(alpha_perturbation_degrees, beta_perturbation_degrees)
perturbation_losses = []
for alpha_perturbation_degree in alpha_perturbation_degrees:
    for beta_perturbation_degree in beta_perturbation_degrees:
        perturbed_mub = perturb_fourier_parameters(factorized_mub, alpha_perturbation_degree, beta_perturbation_degree)
        loss = loss_function(perturbed_mub)
        perturbation_losses.append(loss)

perturbation_losses = np.array(perturbation_losses).reshape(X.shape)
from matplotlib import cm
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, perturbation_losses, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.show()
