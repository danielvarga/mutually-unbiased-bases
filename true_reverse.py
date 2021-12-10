import sys
import numpy as np
from itertools import permutations


TP = 2 * np.pi
TIP = 2j * np.pi
W = np.exp(TIP / 3)


# mub_120_normal.npy
filename = sys.argv[1]
a = np.load(filename)
a *= 6 ** 0.5
b = a[1]
np.set_printoptions(precision=5, suppress=True)


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


def transform(b1, b2):
    ps = list(permutations(range(6)))
    bestdist = 1e9
    bestp1 = None
    bestp2 = None
    for p1 in ps:
        for p2 in ps:
            b1p = b1[p1, :]
            b1p = b1p[:, p2]
            b1p /= b1p[0:1, :]
            b1p /= b1p[:, 0:1]
            dist = np.abs(b1p - b2).sum()
            if dist < bestdist:
                bestdist = dist
                bestp1 = p1
                bestp2 = p2
    # phases are not provided, because i've only encountered
    # F(x,y) -> F(x', y') transformations where both the left and the
    # right diagonals were the identity matrix.
    # but they are easy to get from the returned data.
    # never tested, tbh:
    b1p = b1[bestp1, :]
    b1p = b1p[:, bestp2]
    colphases = b1p[0, :]
    b1p /= b1p[0:1, :]
    rowphases = b1p[:, 0]
    return bestp1, bestp2, bestdist


def the_true_decomposition(b):
    assert np.allclose(np.abs(b), np.ones_like(b)), "b should be sqrt(6) times a Hadamard basis."
    b_orig = b.copy()
    d_left1 = b[:, 0].copy()
    b /= b[:, 0:1]
    d_right1 = b[0, :].copy()
    b /= b[0:1, :]
    b_reconstruct1 = np.diag(d_left1) @ b @ np.diag(d_right1)
    assert np.allclose(b_reconstruct1, b_orig)

    # we are looking for elements that are far from ALL of 1, -1, W, W^2.
    mask = np.abs(b - 1) * np.abs(b - W) * np.abs(b - W ** 2) * np.abs(b + 1)
    # our goal is two find two such elements so that they are not the negative of each other.
    # that has many solutions, but any will suffice as (x, y).
    # this can probably fail when x and y are 6th roots of unity, or very close, but what the heck.
    ind = np.unravel_index(np.argsort(-mask, axis=None), mask.shape)
    # for any i, mask[ind[0][i], ind[1][j]] is the element that's the i-th largest in the whole 2d array.
    # https://stackoverflow.com/a/64338853/383313
    x = b[ind[0][0], ind[1][0]]
    col_x = ind[1][0]
    # let's take another element from x's column that is far from -x.
    # namely, the next possible one in this ordering:
    for i in range(1, len(ind[0])):
        col_y = ind[1][i]
        if col_y == col_x:
            y = b[ind[0][i], ind[1][i]]
            if not np.isclose(-y, x):
                break

    candidate_basis = canonical_fourier(x, y)
    p1, p2, dist = transform(candidate_basis, b)
    b_reconstruct2 = candidate_basis[p1, :]
    b_reconstruct2 = b_reconstruct2[:, p2]
    d_left2 = np.conjugate(b_reconstruct2[:, 0])
    b_reconstruct2 /= b_reconstruct2[:, 0:1]
    d_right2 = np.conjugate(b_reconstruct2[0, :])
    b_reconstruct2 /= b_reconstruct2[0:1, :]
    assert np.allclose(b_reconstruct2, b)

    p_left = np.eye(6)[p1, :]
    p_right = np.eye(6)[:, p2]
    b_reconstruct3 = np.diag(d_left2) @ p_left @ candidate_basis @ p_right @ np.diag(d_right2)
    assert np.allclose(b_reconstruct3, b)

    # b is not to be confused with b_orig. now we have to compose everything to get b_orig:
    d_left = np.diag(d_left1) @ np.diag(d_left2)
    d_right = np.diag(d_right2) @ np.diag(d_right1)
    b_reconstruct4 = d_left @ p_left @ candidate_basis @ p_right @ d_right
    assert np.allclose(b_orig, b_reconstruct4) # b_orig now!
    return d_left, p_left, x, y, p_right, d_right, dist


def phase_to_deg(x):
    return np.angle(x) / np.pi * 180


np.set_printoptions(precision=12, suppress=True, linewidth=100000)

for i in range(1, 4):
    d_left, p_left, x, y, p_right, d_right, dist = the_true_decomposition(b)
    d_l = phase_to_deg(np.diag(d_left))
    d_r = phase_to_deg(np.diag(d_right))
    p_l = np.argmax(p_left, axis=0) # permutation of rows, but not here.
    p_r = np.argmax(p_right, axis=0) # permutation of columns, but not here.
    print("filename", filename, "i", i, "D_l", d_l, "P_l", p_l, "x", phase_to_deg(x), "y", phase_to_deg(y), "P_r", p_r, "D_r", d_r, "distance", dist)
