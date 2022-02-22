import sys
import numpy as np

import sympy
from sympy import symbols, Matrix, Transpose, conjugate, sqrt, cbrt, Rational
from sympy import expand, factor, cancel, nsimplify, simplify, fraction
from sympy.physics.quantum.dagger import Dagger
from sympy.matrices.dense import matrix_multiply_elementwise


TP = 2 * np.pi
TIP = 2j * np.pi
W = np.exp(TIP / 3)

A = 0.42593834 # new name W
B = 0.35506058 # new name U
C = 0.38501704 # new name V
D = 0.40824829
numerical_magic_constants = [36 * c **2 for c in [A, B, C]]

WWsym = sympy.symbols('phi') # 12th root of unity.
magnitudes_sym = sympy.symbols('A B C')


# mub_120_normal.npy
filename = sys.argv[1]
mub = np.load(filename)
mub *= 6 ** 0.5


def angler(x):
    return np.angle(x) * 180 / np.pi


def rot_to_60(p_orig):
    p = p_orig
    while not(0 <= np.angle(p) < np.pi / 3):
        p *= -np.conjugate(W)
    return p


def find_delta(b0, b1):
    WW = np.exp(TIP / 12)
    prod = np.conjugate(b0.T) @ b1
    normed = prod.copy()
    normed /= np.abs(normed)
    vecs = normed.flatten()

    bets = []
    for v1 in vecs:
        for v2 in vecs:
            for v3 in vecs:
                if np.isclose(rot_to_60(v1 * v3 / v2 / v2), 1, atol=1e-4) and not(np.isclose((v2 / v1) ** 12, 1, atol=1e-4)):
                    bets.append((v2, v1))
    alpha, beta = bets[0] # let's just take the first, for now.
    delta = beta / alpha
    alpha = rot_to_60(alpha)
    delta = rot_to_60(delta)
    if angler(delta) > 30:
        delta = WW ** 2 * np.conjugate(delta)
        assert angler(delta) < 30
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

    import matplotlib.pyplot as plt
    vecs = (normed / normed[0, 0]).flatten()
    print(np.sort(angler(vecs)))
    # plt.scatter(vecs.real, vecs.imag) ; plt.show()

    WW = np.exp(TIP / 12)
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
    alpha, delta = find_delta(b0, b1)
    alphasym, deltasym = sympy.symbols(f"alpha{indx}, delta")
    p = reconstruct_product(b0, b1, alpha, delta, alphasym, deltasym)
    return p


def dump_products():
    for (i, j) in [(0, 1), (1, 2), (2, 0)]:
        p = reconstruct_product_full_service(mub[i + 1], mub[j + 1], j)
        print(i, j, p)


np.set_printoptions(precision=3, suppress=True, linewidth=100000)

dump_products()
