import numpy as np
import sympy
from sympy import conjugate
from sympy.physics.quantum import TensorProduct


from base import *

c = np.load("canonized_cubes/canonized_cube_00000.npy")
S_np = c[0, :, :]


Wsym = sympy.symbols('W')
W = np.exp(1j * np.pi * 2/3)

x_var, y_var = sympy.symbols('x y')

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


def circulant_sym(firstrow):
    a, b, c = firstrow
    return sympy.Matrix([[a, b, c], [c, a, b], [b, c, a]])


def szollosi_original_sym(firstrow):
    a, b, c, d, e, f = firstrow
    block1 = circulant_sym([a, b, c])
    block2 = circulant_sym([d, e, f])
    block3 = circulant_sym([conjugate(d), conjugate(f), conjugate(e)])
    block4 = circulant_sym([-conjugate(a), -conjugate(c), -conjugate(b)])
    blockcirculant = sympy.Matrix(sympy.BlockMatrix([[block1, block2], [block3, block4]]))
    return blockcirculant


# print(symbolic_fourier_basis(x_var, y_var)) ; exit()


def F3():
    f = np.ones((3, 3), dtype=np.complex128)
    f[1, 1] = f[2, 2] = W
    f[2, 1] = f[1, 2] = W ** 2
    return f


def F3_sym():
    f = sympy.ones(3, 3)
    f[1, 1] = f[2, 2] = Wsym
    f[2, 1] = f[1, 2] = Wsym ** 2
    return f


def Sk_from_szollosi(S):
    left = np.kron(np.eye(2), F3()) / np.sqrt(3)
    right = np.kron(np.eye(2), np.conj(F3().T)) / np.sqrt(3)
    D = left @ S @ right
    '''
    F = F3() / np.sqrt(3)
    Z = np.zeros((3,3), dtype=complex)
    FB = np.block([
        [F, Z],
        [Z, F]
    ])
    '''
    alpha_ijk = np.zeros((2, 2, 3), dtype=np.complex128)
    for i in range(2):
        for j in range(2):
            for k in range(3):
                alpha_ijk[i, j, k] = D[i * 3 + k, j * 3 + k]
    return alpha_ijk


firstrow = sympy.symbols('a b c d e f')
S_sym = szollosi_original_sym(firstrow)


def Sk_from_szollosi_sym(S_sym):
    left = TensorProduct(sympy.eye(2), F3_sym()) / sympy.sqrt(3)
    right = TensorProduct(sympy.eye(2), sympy.conjugate(F3_sym().T)) / sympy.sqrt(3)
    D = left @ S_sym @ right
    '''
    F = F3_sym() / sympy.sqrt(3)
    Z = sympy.zeros(3, 3)
    FB = sympy.BlockMatrix([
        [F, Z],
        [Z, F]
    ])
    FB = sympy.block_collapse(FB)
    '''
    alpha_ijk = sympy.MutableDenseNDimArray([0] * 12, (2, 2, 3))
    for i in range(2):
        for j in range(2):
            for k in range(3):
                alpha_ijk[i, j, k] = D[i * 3 + k, j * 3 + k]
    return sympy.ImmutableDenseNDimArray(alpha_ijk)


# Sk_sym = Sk_from_szollosi_sym(S_sym) ; print(Sk_sym) ; exit()

# print(Sk_from_szollosi_sym(sympy.eye(6))) ; exit()


#Input: single S matrix
def beta_from_S(S):
    a, b, c, d = S[0, 0], S[0, 1], S[1, 0], S[1, 1]
    beta = np.zeros(4, dtype=np.complex128)
    be2_plus_be3 = d / a
    be2_minus_be3 = c / b
    beta[2] = np.sqrt(be2_plus_be3 * be2_minus_be3)
    beta[3] = np.sqrt(be2_plus_be3 / be2_minus_be3)
    sum_be0_be1 = a
    diff_be0_be1 = b / beta[3]
    beta[0] = sum_be0_be1 + diff_be0_be1
    beta[1] = sum_be0_be1 - diff_be0_be1

    # ZAUNER CONVENTION
    beta[2] = 1 / beta[2]

    if not np.allclose(S_from_beta(beta), S, atol=1e-2):
        beta[2] = - beta[2]
    if not np.allclose(S_from_beta(beta), S, atol=1e-2):
        assert False, 'CANNOT RECONSTRUCT'

    return beta


def beta_from_S_sym(S_sym):
    a, b, c, d = S_sym[0, 0], S_sym[0, 1], S_sym[1, 0], S_sym[1, 1]
    beta = sympy.zeros(4, 1)
    be2_plus_be3 = d / a
    be2_minus_be3 = c / b
    beta[2] = sympy.sqrt(be2_plus_be3 * be2_minus_be3)
    beta[3] = sympy.sqrt(be2_plus_be3 / be2_minus_be3)
    sum_be0_be1 = a
    diff_be0_be1 = b / beta[3]
    beta[0] = sum_be0_be1 + diff_be0_be1
    beta[1] = sum_be0_be1 - diff_be0_be1

    # ZAUNER CONVENTION
    beta[2] = 1 / beta[2]

    print("beta[2] minus unimplemented")
    return beta
    if not np.allclose(S_from_beta_sym(beta), S, atol=1e-2):
        beta[2] = - beta[2]
    if not np.allclose(S_from_beta_sym(beta), S, atol=1e-2):
        assert False, 'CANNOT RECONSTRUCT'

    return beta


def S_from_beta(beta):
    #SZOLLOSI CONVENTION
    # return np.matrix([[(beta[0] + beta[1]) / 2, beta[3] / 2 * (beta[0] - beta[1])],
    #                     [beta[2] / 2 * (beta[0] - beta[1]), beta[2] * beta[3] / 2 * (beta[0] + beta[1])]
    # ])
    # ZAUNER CONVENTION b0=u b1=v b2=x b3=y
    return np.array([[(beta[0] + beta[1]) / 2, beta[3] / 2 * (beta[0] - beta[1])],
                        [1 / beta[2] / 2 * (beta[0] - beta[1]), 1 / beta[2] * beta[3] / 2 * (beta[0] + beta[1])]
    ])


def S_from_beta_sym(beta):
    #SZOLLOSI CONVENTION
    # return np.matrix([[(beta[0] + beta[1]) / 2, beta[3] / 2 * (beta[0] - beta[1])],
    #                     [beta[2] / 2 * (beta[0] - beta[1]), beta[2] * beta[3] / 2 * (beta[0] + beta[1])]
    # ])
    # ZAUNER CONVENTION b0=u b1=v b2=x b3=y
    return sympy.Matrix([[(beta[0] + beta[1]) / 2, beta[3] / 2 * (beta[0] - beta[1])],
                        [1 / beta[2] / 2 * (beta[0] - beta[1]), 1 / beta[2] * beta[3] / 2 * (beta[0] + beta[1])]
    ])


def U_from_Sk(S):
    betas = [beta_from_S(S[:, :, k]) for k in range(3)]
    U = np.array([np.diag([betas[k][l] for k in range(3)]) for l in range(4)])
    print(U.shape)
    return U


def U_from_Sk_sym(Sk):
    betas = [beta_from_S_sym(Sk[:, :, k]) for k in range(3)]
    U = [sympy.zeros(3, 3) for _ in range(4)]
    for l in range(4):
        for k in range(3):
            U[l][k, k] = betas[k][l]
    return U


def E_from_U(U):
    F = F3()
    E1 = np.block([
        [F, U[2] @ F],
        [F, -U[2] @ F]
    ]) / np.sqrt(6)

    E2 = np.block([
        [U[0] @ F, U[0] @ U[3] @ F],
        [U[1] @ F, -U[1] @ U[3] @ F]
    ]) / np.sqrt(6)

    return E1, E2


def E_from_U_sym(U):
    F = F3_sym()
    E1 = sympy.BlockMatrix([
        [F, U[2] @ F],
        [F, -U[2] @ F]
    ]) / sympy.sqrt(6)

    E2 = sympy.BlockMatrix([
        [U[0] @ F, U[0] @ U[3] @ F],
        [U[1] @ F, -U[1] @ U[3] @ F]
    ]) / sympy.sqrt(6)

    return sympy.Matrix(E1), sympy.Matrix(E2)


def szollosi_to_MUB(S):
    #S has to be complex type matrix, Hadamard of Szollosi type
    S_k = Sk_from_szollosi(S)
    print(S_k.shape)
    U = U_from_Sk(S_k)
    print(U.shape)
    E1, E2 = E_from_U(U)
    return S, E1, E2


def szollosi_to_MUB_sym(S):
    #S has to be complex type matrix, Hadamard of Szollosi type
    S_k = Sk_from_szollosi_sym(S)
    U = U_from_Sk_sym(S_k)
    E1, E2 = E_from_U_sym(U)
    return S, E1, E2


# not to be confused with szollosi_original_sym(firstrow), a different parametrization.
def get_szollosi_sym(x, y, u, v):
    S = np.zeros((6, 6), dtype=object)
    S[0, :] = 1
    S[1, :] = [1, x * x * y, x * y * y, x * y / u / v, u * x * y, v * x * y]
    S[2, :] = [1, x / y, x * x * y, x / u, x / v, u * v * x]
    S[3, :] = [1, u * v * x, u * x * y, -1, - u * x * y, - u * v * x]
    S[4, :] = [1, x / u, v * x * y, - x / u, -1, - v * x * y]
    S[5, :] = [1, x / v, x * y / u / v, - x * y / u / v, - x / v, -1]
    return S


def mub_to_cube(fourier1, fourier2):
    n = 6
    a = [sympy.eye(6), fourier1, fourier2]

    c = np.zeros((n, n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                c[i, j, k] = \
                    (sympy.conjugate(a[0][:, i].T) @ a[1][:, j]) * \
                    (sympy.conjugate(a[1][:, j].T) @ a[2][:, k]) * \
                    (sympy.conjugate(a[2][:, k].T) @ a[0][:, i])
    return c


def simplify_roots(expr):
    e = expr.subs(conjugate(Wsym), Wsym ** 2)
    e = e.subs(Wsym ** 3, 1).subs(Wsym ** 4, Wsym).subs(Wsym ** 5, Wsym ** 2).subs(Wsym ** 6, 1)
    return e


def main():
    c = np.load("canonized_cubes/canonized_cube_00000.npy")
    S = c[0, :, :]
    _, E1, E2 = szollosi_to_MUB(S)
    assert np.allclose(np.abs(trans(E1, E2)), 1/6 ** 0.5, atol=1e-6)


def main_sym():
    x, y, u, v = sympy.symbols('x y u v')
    S_sym = get_szollosi_sym(x, y, u, v)

    _, E1_sym, E2_sym = szollosi_to_MUB_sym(S_sym)

    # print(E1_sym)
    # print("=====")
    cube_sym = mub_to_cube(E1_sym, E2_sym)
    print(cube_sym)
    exit()


    E1_sym = simplify_roots(E1_sym)
    E1_sym = sympy.simplify(E1_sym * sympy.sqrt(6))
    print(E1_sym.expand())


# main() ; exit()
main_sym() ; exit()
