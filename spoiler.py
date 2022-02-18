import sys
import numpy as np
import matplotlib.pyplot as plt

import sympy
from sympy import symbols, I, Matrix, Transpose, conjugate, sqrt, Rational, ones
from sympy import expand, factor, cancel, nsimplify, simplify, fraction, lambdify
from sympy.physics.quantum.dagger import Dagger
from sympy.matrices.dense import matrix_multiply_elementwise


# symbol of third root of unity.
Wsym = symbols('W')


def simplify_roots(expr):
    e = expr.subs(conjugate(Wsym), Wsym ** 2)
    e = e.subs(Wsym ** 3, 1).subs(Wsym ** 4, Wsym).subs(Wsym ** 5, Wsym ** 2)
    e = e.subs(Wsym ** 6, 1).subs(Wsym ** 7, Wsym).subs(Wsym ** 8, Wsym ** 2).subs(Wsym ** 9, 1)
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


x_opt_num = np.exp(0.9852j)
t_opt_num = np.exp(1.0094j)


def sym_to_num(f):
    return np.array(subs_roots(f).subs(tsym, t_opt_num).subs(xsym, x_opt_num)).astype(np.complex128)


M1_num = sym_to_num(M1)
M2_num = sym_to_num(M2)
M3_num = sym_to_num(M3)

np.set_printoptions(precision=3, suppress=True)

print(np.abs(np.conjugate(M1_num.T) @ M1_num))
print(np.abs(np.conjugate(M1_num.T) @ M2_num))


def loss_fn(a, b):
    ones6 = ones(6, 1)
    ones66 = ones(6, 6)
    prod = enforce_norm_one(simplify_roots(Dagger(a) @ b), [tsym, xsym])
    mat = matrix_multiply_elementwise(prod, conjugate(prod)) - ones66 / 6
    mat = enforce_norm_one(simplify_roots(mat), [tsym, xsym])
    mat2 = matrix_multiply_elementwise(mat, mat)
    mat2 = enforce_norm_one(simplify_roots(expand(mat2)), [tsym, xsym])
    loss = 1 - (Dagger(ones6) @ mat2 @ ones6)[0, 0] / 5 # Raynal formula (2)
    loss = enforce_norm_one(simplify_roots(expand(loss)), [tsym, xsym])
    return loss


'''
loss = loss_fn(M1, M2)
# loss = loss_fn(N1 / sqrt(6), N2 / sqrt(6))
print(sym_to_num(loss))

mapper = lambdify([xsym, tsym], subs_roots(loss), "numpy")
circle = np.exp(1j * np.linspace(0, 2*np.pi, 90))
torus_x, torus_t = np.meshgrid(circle, circle)
grid = mapper(torus_x, torus_t)
# TODO this it supposed to be 0 and it's not.
plt.imshow(np.imag(grid))
plt.show()
'''


EYE_num = np.eye(6, dtype=np.complex128)
a = np.stack([EYE_num, M1_num, M2_num, M3_num])

np.set_printoptions(precision=3, suppress=True)

for i in range(4):
    prod = np.conjugate(a[i].T) @ a[i]
    # print(i, i, prod)
    assert np.allclose(prod, EYE_num)
    for j in range(i + 1, 4):
        prod = np.conjugate(a[i].T) @ a[j]
        # print(i, j, np.abs(prod))
        if i == 0:
            assert np.allclose(np.abs(prod) ** 2, 1 / 6)
        else:
            print(i, j)
            print(np.abs(prod))

np.save('spoiler.npy', a)

def angler(x):
    return np.angle(x) * 180 / np.pi

# print(repr(np.array(subs_roots(N1).subs(tsym, dummy_t)).astype(np.complex128)))

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

# q_opt = (t * -conj(W)).real
q_opt = (1 - 2 * p2_opt) / p_opt
q2_opt = q_opt ** 2

p, q = sympy.symbols('p q')
P = 8 * p ** 8 + 8 * q ** 2 * p ** 6 - 16 * q ** 3 * p ** 5 \
    + 16 * q * p ** 5 - 16 * q ** 2 * p ** 4 + 8 * q ** 3 * p ** 3 \
    - 7 * p ** 4 - 14 * q * p ** 3 + 8 * q ** 2 * p ** 2 \
    + 2 * p ** 2 + 4 * q * p
print("r numeric", r.evalf())
assert np.isclose(float(r), 0.7199, atol=1e-4)
print("p_opt ** 2 numeric", p_opt.evalf() ** 2, "p_opt aka sin(theta_x^opt) numeric", p_opt.evalf())
assert np.isclose(float(p_opt) ** 2, 0.6946, atol=1e-4)

eq20 = 112 * p_opt ** 6 - 192 * p_opt ** 4 + 111 * p_opt ** 2 -22
print("equation (20)", eq20, "=", eq20.evalf())
assert np.isclose(float(eq20), 0, atol=1e-10)

print("q_opt numeric", q_opt.evalf())

opt = (71 - 12* (1-p2_opt)**2) / 70
print("OPTIMUM", opt.evalf())
assert np.isclose(float(opt), 0.9983, atol=1e-4)

dPdp = sympy.diff(P, p).subs(p, p_opt).subs(q, q_opt)
print("dP(p, q)/dp numeric", dPdp.evalf())
assert np.isclose(float(dPdp), 0)
dPdq = sympy.diff(P, q).subs(p, p_opt).subs(q, q_opt)
print("dP(p, q)/dq numeric", dPdq.evalf())
assert np.isclose(float(dPdq), 0)


x_opt = sqrt(1 - p2_opt) + I * sqrt(p2_opt)
print("x_opt", x_opt.evalf(), "~", x_opt_num)

theta_t_opt_careful = sympy.acos(q_opt) - sympy.pi / 3
t_opt_careful = sympy.exp(I * theta_t_opt_careful)

t_opt = - Wsym * (q_opt + I * sqrt(1 - q2_opt))
print("t_opt_careful", subs_roots(t_opt_careful).evalf(), "t_opt", subs_roots(t_opt).evalf(), "~", t_opt_num)
assert np.isclose(np.complex128(subs_roots(t_opt / t_opt_careful)), 1)

print("t_opt", subs_roots(t_opt.evalf()), "~", t_opt_num)


# updating the numeric values from the 4 significant digit versions to the 15 digit versions:
x_opt_num = np.complex128(x_opt)
t_opt_num = np.complex128(subs_roots(t_opt))

M1_num = sym_to_num(M1)
M2_num = sym_to_num(M2)
M3_num = sym_to_num(M3)

prod12 = subs_roots((Dagger(M1) @ M2).subs(xsym, x_opt).subs(tsym, t_opt))
print(np.abs(np.array(prod12).astype(np.complex128)))

exit()


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
