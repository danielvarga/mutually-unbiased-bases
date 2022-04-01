import sys
import numpy as np
from itertools import permutations, combinations


TP = 2 * np.pi
TIP = 2j * np.pi
W = np.exp(TIP / 3)


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


def graph_to_partition(g):
    g = sorted(g)
    sets = []
    for a, b in g:
        found = False
        for s in sets:
            if a in s:
                s.add(b)
                found = True
                break
        if not found:
            sets.append(set([a, b]))
    return sets


def find_partitions(b):
    atol = 1e-5
    verbose = False
    triangle_graph = []
    biangle_graph = []
    for c1, c2 in list(combinations(range(6), 2)):
        div = b[:, c1] / b[:, c2]
        div /= div[0]
        triangle = np.allclose(div ** 3, 1, atol=atol)
        biangle = np.allclose(div ** 2, 1, atol=atol)
        if triangle:
            triangle_graph.append((c1, c2))
            if verbose:
                print(c1, c2, "T", div ** 3)
        if biangle:
            biangle_graph.append((c1, c2))
            if verbose:
                print(c1, c2, "B", div ** 2)
    bipart = graph_to_partition(triangle_graph)
    tripart = graph_to_partition(biangle_graph)
    if verbose:
        print(bipart, tripart)
    return bipart, tripart


def find_blocks(b):
    transposed = False
    bipart_col, tripart_col = find_partitions(b)
    if len(bipart_col) == 0:
        b = b.T
        transposed = True
    bipart_col, tripart_col = find_partitions(b)
    assert len(bipart_col) == 2 and len(bipart_col[0]) == 3 and len(bipart_col[1]) == 3, f"bad bipart {bipart_col}"
    bipart_row, tripart_row = find_partitions(b.T)
    assert len(tripart_row) == 3 and all(len(tripart_row[i]) == 2 for i in range(3)), f"bad tripart {tripart_row}"


def hadamard_cube(a, pad_with_id=True):
    n = 6
    if pad_with_id:
        a = np.stack([np.eye(n, dtype=np.complex128), a[0], a[1]])
    assert a.shape == (3, n, n)
    # vectorize maybe? who cares?
    c = np.zeros((n, n, n), dtype=np.complex128)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                c[i, j, k] = \
                    (np.conjugate(a[0, :, i]) @ a[1, :, j]) * \
                    (np.conjugate(a[1, :, j]) @ a[2, :, k]) * \
                    (np.conjugate(a[2, :, k]) @ a[0, :, i])
    return 6 * c


def phase_to_deg(x):
    return np.angle(x) / np.pi * 180


def verify_hadamard(b):
    n = len(b)
    assert np.allclose(np.abs(b) ** 2, 1 / n, atol=1e-4)
    prod = np.conjugate(b.T) @ b
    assert np.allclose(prod, np.eye(n), atol=1e-4)


def gently_verify_hadamard(b):
    n = len(b)
    print(np.abs(b) ** 2 * n - 1)
    prod = np.conjugate(b.T) @ b
    print(prod - np.eye(n))


def angler(x):
    return np.angle(x) * 180 / np.pi
