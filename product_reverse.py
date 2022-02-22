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
WW = np.exp(TIP / 12)

A = 0.42593834 # new name W
B = 0.35506058 # new name U
C = 0.38501704 # new name V
D = 0.40824829
numerical_magic_constants = [36 * c **2 for c in [A, B, C]]

Wsym, WWsym = sympy.symbols('omega phi') # 3rd and 12th roots of unity.
magnitudes_sym = sympy.symbols('A B C')

BOARD = [(i, j) for i in range(6) for j in range(6)]


# mub_120_normal.npy
filename = sys.argv[1]
mub = np.load(filename)
mub *= 6 ** 0.5
mub = mub[1:] # remove identity

def angler(x):
    return np.angle(x) * 180 / np.pi


value_dict = {Wsym: W, WWsym: WW,
    magnitudes_sym[0]: A, magnitudes_sym[1]: B, magnitudes_sym[2]: C}

def sym_to_num(formula):
    f = formula
    for sym, num in value_dict.items():
        f = f.subs(sym, num)
    try:
        a = np.array(f, dtype=np.complex128)
    except:
        print("failed to fully evaluate", formula)
        print("still variables left in", f)
        raise
    return np.squeeze(a)



# between -30 and 30 actually, so that 60-divisible angles
# are stably land around 0.
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


# a slower version of identify_alpha_delta(b0, b1)
def identify_alpha_delta_obsolete(b0, b1):
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
    assert np.allclose(sym_to_num(alphasym * normed_prod_sym), normed, atol=1e-4)

    msquared = np.abs(prod) ** 2
    masks = [np.isclose(msquared, numerical_magic_constants[i]).astype(int) for i in range(3)]
    masks = np.array(masks)
    magnitude_matrix_sym = sympy.ones(6, 6)
    for i in range(6):
        for j in range(6):
            element_index = np.argmax(masks[:, i, j])
            magnitude_matrix_sym[i, j] *= magnitudes_sym[element_index]
    assert np.allclose(sym_to_num(6 * magnitude_matrix_sym), np.abs(prod), atol=1e-4)
    final_result = matrix_multiply_elementwise(normed_prod_sym, magnitude_matrix_sym)
    assert np.allclose(sym_to_num(6 * alphasym * final_result), prod, atol=1e-4)
    return final_result


# BEWARE, this is modifying global state!
# at least it does not overwrite it without checking.
def reconstruct_product_full_service(b0, b1, indx):
    alpha, delta = identify_alpha_delta(b0, b1)

    alphasym, deltasym = sympy.symbols(f"alpha{indx}, delta")

    assert alphasym not in value_dict
    value_dict[alphasym] = alpha
    if deltasym in value_dict:
        assert np.isclose(value_dict[deltasym], delta, atol=1e-4)
    value_dict[deltasym] = delta

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


mub = alpha_removal(mub)

def dump_products():
    for (i, j) in [(0, 1), (1, 2), (2, 0)]:
        p = reconstruct_product_full_service(mub[i], mub[j], j)
        print(i, j, p)


np.set_printoptions(precision=3, suppress=True, linewidth=100000)

dump_products()
