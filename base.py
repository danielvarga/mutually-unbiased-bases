import sys
import numpy as np
from itertools import permutations, combinations


TP = 2 * np.pi
TIP = 2j * np.pi
W = np.exp(TIP / 3)


def dag(b):
    return np.conjugate(b.T)


def trans(b1, b2):
    return np.conjugate(b1.T) @ b2


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


def dephase(bb):
    b = bb.copy()
    b /= b[0, :]
    b /= b[:, 0][:, np.newaxis]
    return b


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
    return [frozenset(s) for s in sets]


def partition_to_coloring(sets):
    c = np.zeros(6, dtype=int)
    for color, s in enumerate(sets):
        for e in s:
            c[e] = color
    return c


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


# For M complex Hadamard, returns M dephased.
def complex_dephase(M):
    M = M.copy()
    Dr = M[0, :].copy()
    M /= Dr
    Dl = M[:, 0][:, np.newaxis].copy()
    M /= Dl
    return M, Dl, Dr


# If M is equal to a canonical fourier, returns its parameters, else None.
def get_complex_fourier(M):
    x = M[4,3]
    y = M[2,3]
    if np.allclose(M, canonical_fourier(x, y), atol=1e-3):
        return x, y
    else:
        return None


# NOTE: partition (list of frozensets), not coloring!
def is_compatible(perm, partition):
    def apply(perm, partition):
        partition2 = []
        for s in partition:
            s2 = frozenset(perm[i] for i in s)
            partition2.append(s2)
        return partition2
    partition2 = apply(perm, partition)
    return frozenset(partition) == frozenset(partition2)


def test_is_compatible():
    partition = [frozenset([0, 4]), frozenset([2, 3]), frozenset([5, 1])]
    perm = [4, 5, 3, 2, 0, 1]
    assert is_compatible(perm, partition)
    perm = [4, 5, 3, 2, 1, 0]
    assert not is_compatible(perm, partition)
    perm = [3, 5, 4, 0, 2, 1]
    assert is_compatible(perm, partition)
    partition = [frozenset([0, 1, 2]), frozenset([3, 4, 5])]
    perm = [4, 5, 3, 1, 0, 2]
    assert is_compatible(perm, partition)
    perm = [4, 5, 1, 3, 0, 2]
    assert not is_compatible(perm, partition)


# test_is_compatible() ; exit()


def get_canonizer(b, bipart_col, tripart_row):
    ps = list(permutations(range(6)))
    print("bipart_col", bipart_col, "tripart_row", tripart_row)
    col_perms = ps # [perm for perm in ps if is_compatible(perm, bipart_col)]
    row_perms = ps # [perm for perm in ps if is_compatible(perm, tripart_row)]
    print(len(col_perms), len(row_perms))
    for col_perm in col_perms:
        col_perm_m = np.eye(6)[:, col_perm]
        for row_perm in row_perms:
            row_perm_m = np.eye(6)[row_perm, :]
            print(col_perm, row_perm)
            b_permuted = row_perm_m @ b @ col_perm_m
            b_dephased, Dl, Dr = complex_dephase(b_permuted)
            p = get_complex_fourier(b_dephased)
            if p != None:
                g = {'d_l' : Dl, 'd_r' : Dr, 'p_l' : row_perm, 'p_r' : col_perm, 'x' : p[0], 'y' : p[1]}
                # print('x = {}, y = {}'.format(angler(g['x']), angler(g['y'])))
                # print(angler(canonical_fourier(g['x'], g['y'])))
                return g
    return None


def is_equivalent(b1, b2):
    ps = list(permutations(range(6)))
    b2_dephased, _, _ = complex_dephase(b2)
    for col_perm in ps:
        col_perm_m = np.eye(6)[:, col_perm]
        for row_perm in ps:
            row_perm_m = np.eye(6)[row_perm, :]
            b1_permuted = row_perm_m @ b1 @ col_perm_m
            b1_dephased, _, _ = complex_dephase(b1_permuted)
            if np.allclose(b1_dephased, b2_dephased, atol=1e-4):
                return True
    return False


def find_blocks(b, allow_transposal=False):
    is_transposed = False
    bipart_col, tripart_col = find_partitions(b)
    if allow_transposal and len(bipart_col) == 0:
        b = b.T
        is_transposed = True
    bipart_col, tripart_col = find_partitions(b)
    if not(len(bipart_col) == 2 and len(bipart_col[0]) == 3 and len(bipart_col[1]) == 3):
        return None
    bipart_row, tripart_row = find_partitions(b.T)
    if not(len(tripart_row) == 3 and all(len(tripart_row[i]) == 2 for i in range(3))):
        return None
    return bipart_col, tripart_row, is_transposed


def hadamard_cube(a, pad_with_id=True):
    n = 6
    if pad_with_id and len(a) == 2:
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
