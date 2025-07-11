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


def switch_perm_phase(pm, phases):
    matrix = pm @ np.diag(phases)
    phases2 = np.sum(matrix, axis=1)
    matrix2 = np.diag(1 / phases2) @ matrix
    assert np.allclose(matrix, np.diag(phases2) @ matrix2)
    assert np.allclose(matrix2 * (matrix2 - 1), 0)
    pm2 = np.around(matrix2).real.astype(int)
    return phases2, pm2


def switch_phase_perm(phases, pm):
    phases2, pm2 = switch_perm_phase(pm.T, phases)
    return pm2.T, phases2


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


def find_partitions(b, atol=1e-4):
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
def complex_dephase(M, row=0, col=0):
    M = M.copy()
    Dr = M[row, :].copy()
    M /= Dr
    Dl = M[:, col][:, np.newaxis].copy()
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


def is_diagonal(x):
    return np.allclose(x, np.diag(np.diagonal(x)))

def create_action(dl, pl, pr, dr):
    assert is_diagonal(dl)
    assert is_diagonal(dr)
    g = {'d_l' : dl,
          'p_l' : pl,
          'd_r' : dr,
          'p_r' : pr
          }
    return g


def create_generalized_fourier(dl, pl, x, y, pr, dr):
    assert is_diagonal(dl)
    assert is_diagonal(dr)
    g = {'d_l' : dl,
          'p_l' : pl,
          'd_r' : dr,
          'p_r' : pr,
          'x': x,
          'y': y
          }
    return g


def invert(g):
    g_inv = {'d_l' : g['p_l'].T @ g['d_l'].conjugate() @ g['p_l'],
            'p_l' : g['p_l'].T,
            'd_r' : g['p_r'] @ g['d_r'].conjugate() @ g['p_r'].T,
            'p_r' : g['p_r'].T}
    return g_inv


def multiply(g, h):
    gh = {'d_l' : np.multiply(g['d_l'], g['p_l'] @ h['d_l'] @ g['p_l'].T),
          'p_l' : g['p_l'] @ h['p_l'],
          'd_r' : np.multiply(g['p_r'].T @ h['d_r'] @ g['p_r'], g['d_r']),
          'p_r' : h['p_r'] @ g['p_r']}
    return gh


def act(g, x):
    return g['d_l'] @ g['p_l'] @ x @ g['p_r'] @ g['d_r']


def rebuild_from_canon(g):
    F = canonical_fourier(g['x'], g['y'])
    return act(g, F)


def get_canonizer(M):
    ps = list(permutations(range(6)))
    for perm1 in ps:
        perm1m = np.eye(6)[perm1, :]
        for perm2 in ps:
            perm2m = np.eye(6)[perm2, :]
            M_permuted = perm1m @ M @ perm2m
            M_dephased, Dl, Dr = complex_dephase(M_permuted)
            p = get_complex_fourier(M_dephased)
            if p != None:
                g = {'d_l' : np.diag(np.conj(Dl[:, 0])), 'd_r' : np.diag(np.conj(Dr)), 'p_l' : perm1m, 'p_r' : perm2m
                    }
                g = invert(g)
                g.update({'x' : p[0], 'y' : p[1]})
                return g
    return None

def get_canonizer_bad(b, bipart_col=None, tripart_row=None):
    ps = list(permutations(range(6)))
    col_perms = ps # [perm for perm in ps if is_compatible(perm, bipart_col)]
    row_perms = ps # [perm for perm in ps if is_compatible(perm, tripart_row)]
    for col_perm in col_perms:
        col_perm_m = np.eye(6)[:, col_perm]
        for row_perm in row_perms:
            row_perm_m = np.eye(6)[row_perm, :]
            b_permuted = row_perm_m @ b @ col_perm_m
            b_dephased, Dl, Dr = complex_dephase(b_permuted)
            p = get_complex_fourier(b_dephased)
            if p != None:
                g = {'d_l' : Dl, 'd_r' : Dr, 'p_l' : row_perm, 'p_r' : col_perm, 'x' : p[0], 'y' : p[1]}
                # print('x = {}, y = {}'.format(angler(g['x']), angler(g['y'])))
                # print(angler(canonical_fourier(g['x'], g['y'])))
                return g
    return None


# equivalent without permutations
def is_phase_equivalent(b1, b2, atol=1e-4):
    b1, _, _ = complex_dephase(b1)
    b2, _, _ = complex_dephase(b2)
    return np.allclose(b1, b2, atol=atol)


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


# Tries to phase M1 into M2. If possible, returns the phases, else None.
def rephase(M1, M2, atol=1e-4):
    M1 = M1.copy()
    Dr = M1[0, :].copy() / M2[0,:].copy()
    M1 /= Dr
    Dl = M1[:, 0][:, np.newaxis].copy() / M2[:, 0][:, np.newaxis].copy()
    M1 /= Dl
    if np.allclose(M1, M2, atol=atol):
        return Dl, Dr
    else:
        return None


# Checks if M1, M2 Hadamard matrices are equivalent.
def equivalent_hadamard_brute_force(M1, M2, atol=1e-4):
    ps = list(permutations(range(6)))
    for perm1 in ps:
        perm1m = np.eye(6)[perm1, :]
        for perm2 in ps:
            perm2m = np.eye(6)[perm2, :]
            M1_permuted = perm1m @ M1 @ perm2m
            p = rephase(M1_permuted, M2, atol=atol)
            if p != None:
                g = {'d_l' : p[0], 'd_r' : p[1], 'p_l' : perm1, 'p_r' : perm2}
                return g
    return None


def lex(b):
    return np.lexsort(b.T[::-1])


def canonically_permute(b):
    i = list(range(6))
    changed = True
    row_perm = np.arange(6)
    col_perm = np.arange(6)
    while changed:
        changed = False
        perm = lex(b)
        b = b[perm]
        row_perm = row_perm[perm]
        if np.any(perm != i):
            changed = True
        perm = lex(b.T)
        b = b.T[perm].T
        col_perm = col_perm[perm]
        if np.any(perm != i):
            changed = True
    return b, row_perm, col_perm


def equivalent_hadamard(b1, b2, atol=1e-4):
    b1, d1l, d1r = complex_dephase(b1)
    an1, row_perm1, col_perm1 = canonically_permute(np.angle(b1) + np.pi)
    for row in range(6):
        for col in range(6):
            b2dephased, d2l, d2r = complex_dephase(b2, row=row, col=col)
            an2, row_perm2, col_perm2 = canonically_permute(np.angle(b2dephased) + np.pi)
            if np.allclose(np.exp(1j * an1), np.exp(1j * an2), atol=atol):
                # TODO should figure out the transformation.
                return True
    return None


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


# does not and cannot arrange within blocks. (2x2x2 x 6x6 choices)
# does not and cannot arrange blocks either. (2 x 6 choices)
# but at least after arrangement it obeys the block partition.
def arrange_blocks(b):
    result = find_blocks(b, allow_transposal=False)
    assert result is not None
    bipart_col, tripart_row, is_transposed = result
    col_perm = list(bipart_col[0]) + list(bipart_col[1])
    row_perm = list(tripart_row[0]) + list(tripart_row[1]) + list(tripart_row[2])
    b = b[:, col_perm]
    b = b[row_perm, :]
    result = find_blocks(b, allow_transposal=False)
    return b


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


def counts(c):
    N = 100000
    bins = (N * np.angle(c)).astype(int)
    bins = np.abs(bins)
    vals, cnts = np.unique(bins.flatten(), return_counts=True)
    return cnts


# For an Hadamard cube make the first direction the distinguished one.
def distinguished_permute(H):
    for p in range(3):
        # print(len(count_elements(H[0, :, :])))
        if len(counts(H[0, :, :])) == 12:
            return H
        else:
            H = np.transpose(H, axes=[1, 2, 0])
    return None
    # print(H)
    # assert False, 'FAULTY CUBE.'


def circulant(firstrow):
    a, b, c = firstrow
    return np.array([[a, b, c], [c, a, b], [b, c, a]])


def szollosi_original(firstrow):
    a, b, c, d, e, f = firstrow
    block1 = circulant([a, b, c])
    block2 = circulant([d, e, f])
    block3 = circulant([np.conjugate(d), np.conjugate(f), np.conjugate(e)])
    block4 = circulant([-np.conjugate(a), -np.conjugate(c), -np.conjugate(b)])
    blockcirculant = np.block([[block1, block2], [block3, block4]])
    return blockcirculant


def phase_to_deg(x):
    return np.angle(x) / np.pi * 180


def verify_hadamard(b, atol=1e-4):
    n = len(b)
    assert np.allclose(np.abs(b) ** 2, 1 / n, atol=atol)
    prod = np.conjugate(b.T) @ b
    assert np.allclose(prod, np.eye(n), atol=atol)


def verify_unbiased(b1, b2, atol=1e-4):
    n = len(b1)
    prod = trans(b1, b2)
    assert np.allclose(np.abs(prod), n ** -0.5, atol=atol)


def verify_mub(a, atol=1e-4):
    m, n, _ = a.shape

    for b in a:
        prod = np.conjugate(b.T) @ b
        assert np.allclose(prod, np.eye(n), atol=atol)
    for i in range(m):
        for j in range(i + 1, m):
            verify_unbiased(a[i], a[j], atol=atol)


def gently_verify_hadamard(b):
    n = len(b)
    print(np.abs(b) ** 2 * n - 1)
    prod = np.conjugate(b.T) @ b
    print(prod - np.eye(n))


def verify_sum(v, atol=1e-4):
    assert np.isclose(v.sum(), 1, atol=atol)


def verify_phase_equivalence(b1, b2, atol=1e-4):
    assert is_phase_equivalent(b1, b2, atol=atol)


# equivalent without permutations
def verify_phase_equivalence_overkill(b1, b2, atol=1e-4):
    rat = b2 / b1
    u, s, vh = np.linalg.svd(rat)
    assert np.isclose(s[0], 6)
    uu = u[:, 0] * 6 ** 0.5
    vv = vh[0, :] * 6 ** 0.5
    assert np.allclose(np.outer(uu, vv), rat, atol=atol)
    assert np.allclose(np.diag(uu) @ b1 @ np.diag(vv), b2, atol=atol)
    return uu, vv


def verify_cube_properties(c, atol=1e-4):
    n = c.shape[0]
    for i in range(n):
        # print("verifying 2D slices", i)
        verify_hadamard(c[i, :, :], atol=atol)
        verify_hadamard(c[:, i, :], atol=atol)
        verify_hadamard(c[:, :, i], atol=atol)

    for i in range(n):
        for j in range(n):
            # print("verifying 1D slices", i, j)
            verify_sum(c[i, j, :], atol=atol)
            verify_sum(c[:, i, j], atol=atol)
            verify_sum(c[j, :, i], atol=atol)

    for i in range(n):
        # print("verifying equivalence of parallel slices", i)
        b1 = c[0, :, :]
        b2 = c[i, :, :]
        verify_phase_equivalence(b1, b2, atol=atol)
        b1 = c[:, 0, :]
        b2 = c[:, i, :]
        verify_phase_equivalence(b1, b2, atol=atol)
        b1 = c[:, :, 0]
        b2 = c[:, :, i]
        verify_phase_equivalence(b1, b2, atol=atol)


def slic(c, direction, coord):
    if direction == 0:
        return c[coord, :, :]
    elif direction == 1:
        return c[:, coord, :]
    elif direction == 2:
        return c[:, :, coord]
    assert False


# Sz-slices come in pairs. This function creates one from the other.
def conjugate_pair(sx):
    b00 = sx[:3, :3]
    b01 = sx[:3, 3:]
    b10 = sx[3:, :3]
    b11 = sx[3:, 3:]
    sx1 = np.block([[dag(b11), dag(b10)], [dag(b01), dag(b00)]])
    return sx1


def projection_to_vector(P):
    u = P[:, 0]
    return u / np.abs(u) / 6 ** 0.5

    u, s, vh = np.linalg.svd(P)
    return u[:, 0]


def vector_to_projection(v):
    return v[:, None] @ v[None, :]


def cube_to_mub(H):
    A = np.eye(6, dtype=np.complex128)
    B = H[:, :, 0]
    Q = np.array([vector_to_projection(B[:, i]) for i in range(6)])
    R = []
    for j in range(6):
        r = np.zeros((6, 6), dtype=np.complex128)
        for k in range(6):
            r[k, :] = np.conj(H[k, :, j]) @ Q[:, k, :]
        R.append(r)

    r_candidate = [projection_to_vector(r) for r in R]
    C = np.array(r_candidate).T
    return np.array([A, B, C])


def cube_to_mub_simplified(H):
    A = np.eye(6, dtype=np.complex128)
    B = H[:, :, 0]
    C = np.zeros((6, 6), dtype=np.complex128)
    for j in range(6):
        for k in range(6):
            C[k, j] = np.sum(np.conj(H[k, :, j]) * B[k, :] * B[0, :])
    # yeah we could broadcast this, but i wanted it super transparent:
    for j in range(6):
        # np.abs(C[:, j]) is a constant vector.
        C[:, j] /= np.abs(C[:, j]) * 6 ** 0.5
    return np.array([A, B, C])


def angler(x):
    return np.angle(x) * 180 / np.pi


def visualize_clusters(c, group_conjugates=True):
    n = 6
    N = 10000
    bins = (N * np.angle(c)).astype(int)
    signs = np.sign(bins)
    assert np.all(signs !=0)
    bins = np.abs(bins)
    vals, cnts = np.unique(bins.flatten(), return_counts=True)

    # for a full cube every color appears 6 times.
    #   (or 3 times if conjugates are not grouped)
    # for a single slice it's either all distinct or
    # every color appears 3 times.
    # for sporadic Fouriers (6th roots of unity) it's 12 or 6 per color.
    assert np.all(cnts == 6) or np.all(cnts == 1) or np.all(cnts == 3) \
        or sorted(cnts) == [6, 6, 12, 12]
    dists = bins[..., None] - vals[None, :]
    close = np.isclose(dists, 0)
    which = np.argmax(close, axis=-1)
    which += 1
    which *= signs
    if group_conjugates:
        which = np.abs(which)
    print(len(np.unique(which.flatten())))
    return which


def tao():
    TIP = np.pi * 2j
    w = np.exp(TIP / 3)
    w2 = w ** 2
    T = np.array([
       [1, 1,  1,  1,  1,  1],
       [1, 1,  w,  w,  w2, w2],
       [1, w,  1,  w2, w2, w],
       [1, w,  w2, 1,  w,  w2],
       [1, w2, w2, w,  1,  w],
       [1, w2, w,  w2, w,  1]])
    return T

