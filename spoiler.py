import sys
import numpy as np
import matplotlib.pyplot as plt

import sympy
from sympy import symbols, I, Matrix, Transpose, conjugate, sqrt, Rational
from sympy import expand, factor, cancel, nsimplify, simplify, fraction
from sympy.physics.quantum.dagger import Dagger
from sympy.matrices.dense import matrix_multiply_elementwise


# symbol of third root of unity.
Wsym = symbols('W')


def simplify_roots(expr):
    e = expr.subs(conjugate(Wsym), Wsym ** 2)
    e = e.subs(Wsym ** 3, 1).subs(Wsym ** 4, Wsym).subs(Wsym ** 5, Wsym ** 2).subs(Wsym ** 6, 1)
    return e


def subs_roots(f):
    f = f.subs(Wsym, - Rational(1, 2) + 1j * sqrt(3) * Rational(1, 2))
    # f = f.subs(WWsym, Rational(1, 2) * 1j + sqrt(3) * Rational(1, 2))
    f = expand(f)
    return f


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


Wsym = sympy.symbols('W')
xsym, tsym = sympy.symbols('x t')


Z = Matrix([[1, 0], [0, -1]])
X = Matrix([[conjugate(xsym), 0], [0, xsym]])
F2 = Matrix([[1, 1], [1, -1]])
T = Matrix([[1, Wsym * tsym ** 2], [1, -Wsym * tsym ** 2]])

def build(minis):
    X = sympy.zeros(6, 6)
    for i in range(3):
        for j in range(3):
            X[i*2: (i+1)*2, j*2: (j+1)*2] = minis[i][j]
    return X

ZERO = sympy.zeros(2, 2)
X1 = build([
    [X, ZERO, ZERO],
    [ZERO,  I * conjugate(Wsym) * tsym * Z @ conjugate(X) ** 2, ZERO],
    [ZERO, ZERO, X]])
N1 = build([
    [F2, F2, F2],
    [F2, Wsym * F2, conjugate(Wsym) * F2],
    [T, conjugate(Wsym) * T, Wsym * T]])
M1 = X1 @ N1 / sqrt(6)

N2 = build([
    [F2, F2, F2],
    [T, Wsym * T, conjugate(Wsym) * T],
    [T, conjugate(Wsym) * T, Wsym * T]])
M2 = N2 / sqrt(6)

X3 = build([
    [conjugate(X), ZERO, ZERO],
    [ZERO, conjugate(Wsym) * conjugate(X), ZERO],
    [ZERO, ZERO, - I * tsym * Z @ X ** 2]])
N3 = build([
    [F2, F2, F2],
    [T, Wsym * T, conjugate(Wsym) * T],
    [F2, conjugate(Wsym) * F2, Wsym * F2]])
M3 = X3 @ N3 / sqrt(6)

def dump(m):
    return nsimplify(simplify(enforce_norm_one(subs_roots(m), [xsym, tsym])))

print("self-products")
print(dump(Dagger(M1) @ M1))
print(dump(Dagger(M2) @ M2))
print(dump(Dagger(M3) @ M3))

'''
print("determinants")
print(dump(M1.det()), dump(M2.det()), dump(M3.det()))
'''

# formula (21)
r = sympy.root(21 * sqrt(3) - 36, 3)

# p_opt aka sqrt(p2_opt) = x.imag
p2_opt = (3 + 16 * r - r ** 2) / 28 / r
p_opt = sqrt(p2_opt)

# q_opt = (t * W).real
q_opt = (1 - 2 * p2_opt) / p_opt
q2_opt = q_opt ** 2

p, q = sympy.symbols('p q')
P = 8 * p ** 8 + 8 * q ** 2 * p ** 6 - 16 * q ** 3 * p ** 5 \
    + 16 * q * p ** 5 - 16 * q ** 2 * p ** 4 + 8 * q ** 3 * p ** 3 \
    - 7 * p ** 4 - 14 * q * p ** 3 + 8 * q ** 2 * p ** 2 \
    + 2 * p ** 2 + 4 * q * p
print("p_opt numeric", p_opt.evalf())
print("q_opt numeric", q_opt.evalf())

opt = (71 - 12* (1-p_opt.evalf()**2)**2)/70
print("OPTIMUM", opt)

print("P", P.subs(p, p_opt).subs(q, q_opt).evalf())

x_opt = sqrt(1 - p2_opt) + I * sqrt(p2_opt)
t_opt = (sqrt(q2_opt) + I * sqrt(1 - q2_opt)) * Wsym
# evalf does not get rid of I, still a sympy formula
x_opt_num = np.complex128(x_opt.evalf())
t_opt_num = np.complex128(subs_roots(t_opt).evalf())


def our_closeness(a, b):
    return np.sum(np.abs(a - b) ** 2)


def their_closeness(a):
    return np.sum((np.abs(a) ** 2 - 1/6) ** 2)


# numpy code reimplementing the tf code search.py:loss_fn()
def our_loss_function(mub):
    terms = []
    for u in mub:
        prod = np.conjugate(u.T) @ u
        assert np.allclose(prod, np.eye(6))

    target = 1 / 6 ** 0.5
    for i in range(3):
        for j in range(i + 1, 3):
            # / 6 because what we call basis is 6**2 times what search.py calls basis.
            prod = np.conjugate(mub[i].T) @ mub[j]
            terms.append(our_closeness(np.abs(prod), target))
    return sum(terms)


def their_loss_function(mub):
    terms = []
    for u in mub:
        prod = np.conjugate(u.T) @ u
        assert np.allclose(prod, np.eye(6))

    for i in range(3):
        for j in range(i + 1, 3):
            # / 6 because what we call basis is 6**2 times what search.py calls basis.
            prod = np.conjugate(mub[i].T) @ mub[j]
            terms.append(their_closeness(prod))
    return sum(terms) / 5 - 1


def angler(x):
    return np.angle(x) / np.pi * 180


print("x_opt", x_opt_num, angler(x_opt_num))
print("t_opt", t_opt_num, angler(t_opt_num))

def sym_to_num(f):
    return np.array(subs_roots(f.subs(xsym, x_opt).subs(tsym, t_opt)), dtype=np.complex128)

M1_num = sym_to_num(M1)
M2_num = sym_to_num(M2)
M3_num = sym_to_num(M3)

MUB_num = np.stack([np.eye(6, dtype=np.complex128), M1_num, M2_num, M3_num])

np.save("mub_spoiler.npy", MUB_num)

print("our loss", our_loss_function(MUB_num[1:]))
print("their loss", their_loss_function(MUB_num[1:]))

np.set_printoptions(precision=12, suppress=True)
print(np.abs(np.conjugate(MUB_num[1]) @ MUB_num[2]))
print(np.abs(np.conjugate(MUB_num[2]) @ MUB_num[3]))
print(np.abs(np.conjugate(MUB_num[3]) @ MUB_num[0]))

exit()

prod = Dagger(M1) @ M2

magsquared = matrix_multiply_elementwise(prod, conjugate(prod))
print("product magnitudes squared delta", dump(magsquared - sympy.ones(6, 6) * magsquared[0, 1]))
