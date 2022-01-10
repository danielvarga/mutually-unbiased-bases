import sys
import numpy as np
import matplotlib.pyplot as plt

import sympy
from sympy import symbols, Matrix, Transpose, conjugate, sqrt, Rational
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


WWsym = sympy.symbols('WW') # 12th root of unity.
magnitudes_sym = sympy.symbols('A B C')
xvar, yvar = symbols('x y')
alphasym, deltasym = sympy.symbols('alpha delta')
left_phases_var = Matrix([[sympy.symbols(f'p1{i}') for i in range(6)]]) # row vector


def sym_to_num(formula):
    f = formula.subs(Wsym, W).subs(WWsym, np.exp(TIP/12))
    f = f.subs(xvar, x)
    # f = f.subs(BperAvar, B/A).subs(CperAvar, C/A)
    f = f.subs(magnitudes_sym[0], A).subs(magnitudes_sym[1], B).subs(magnitudes_sym[2], C)
    f = f.subs(alphasym, alpha).subs(deltasym, delta)
    # TODO we don't substitute d_2.
    for i in range(6):
        f = f.subs(left_phases_var[i], d_1[i])
    try:
        a = np.array(f, dtype=np.complex128)
    except:
        print("failed to fully evaluate", formula)
        print("still variables left in", f)
        raise
    return np.squeeze(a)


alpha = 0.8078018037463457+0.589454193185654j
delta = -0.12834049204788406+0.9917301639563593j

