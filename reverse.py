# the goal is to re-create mub_120_normal.npy as parametrized matrix

import sys
import numpy as np


TP = 2 * np.pi
TIP = 2j * np.pi
W = np.exp(TIP / 3)


# mub_120_normal.npy
filename = sys.argv[1]
a = np.load(filename)
a *= 6 ** 0.5
b = a[1]
np.set_printoptions(precision=5, suppress=True)

# 3 goes to 5 so that the W block is consecutive
b = b[[0, 1, 2, 4, 5, 3]]

x = b[1:5, 0].copy()
y = b[1:5, 3].copy()
z = b[5, 0]

b[1:5, :3] /= x[:, None]
b[1:5, 3:] /= y[:, None]
b[5, :] /= z
print(np.angle(b) / TP * 360)

print("x", np.angle(x)  / TP * 360)
print("y", np.angle(y)  / TP * 360)


def verify_constraints(x, y):
    a = np.angle(x)
    b = np.angle(y)

    c30_angle = np.mod(a[3] - a[0] - b[3] + b[0], TP)
    c12_angle = np.mod(a[1] - a[2] - b[1] + b[2], TP)
    c01_angle = np.mod(a[0] - a[1] - b[0] + b[1], TP)
    c23_angle = np.mod(a[2] - a[3] - b[2] + b[3], TP)
    c02_angle = np.mod(a[0] - a[2] - b[0] + b[2], TP)
    c13_angle = np.mod(a[1] - a[3] - b[1] + b[3], TP)

    def piclose(x):
        return np.isclose(x, np.pi, atol=1e-4)

    r12_30 = piclose(c30_angle) and piclose(c12_angle)
    r01_23 = piclose(c01_angle) and piclose(c23_angle)
    r02_13 = piclose(c02_angle) and piclose(c13_angle)
    assert r12_30 or r01_23 or r02_13
    print(c30_angle, c12_angle, c01_angle, c23_angle, c02_angle, c13_angle)

    if not r12_30:
        print("it has the fourier structure, but the rows are permuted differently")

    print("verified constraints")

    # this code only works with 12-30 separated rows,
    # which is true for mub_120_normal.npy but not true for optimum.npy:
    '''
    c1 = np.conj(x[0]) * x[3] + np.conj(y[0]) * y[3]
    c2 = np.conj(x[1]) * x[2] + np.conj(y[1]) * y[2]
    assert np.isclose(c1, 0)
    assert np.isclose(c2, 0)

    c1_angle = np.mod(a[3] - a[0] - b[3] + b[0] + np.pi, TP)
    c2_angle = np.mod(a[2] - a[1] - b[2] + b[1] + np.pi, TP)
    assert np.isclose(c1_angle, 0) or np.isclose(c1_angle, TP)
    assert np.isclose(c2_angle, 0) or np.isclose(c2_angle, TP)
    c1_angle = np.mod(a[3] - a[0] - b[3] + b[0], TP)
    c2_angle = np.mod(a[2] - a[1] - b[2] + b[1], TP)
    assert np.isclose(c1_angle, np.pi)
    assert np.isclose(c2_angle, np.pi)
    print("verified constraints")
    '''

verify_constraints(x, y)


def create_random():
    a = np.zeros(4, dtype=np.complex128)
    b = np.random.uniform(size=4)
    b[3] = b[0] + 1/2
    b[2] = b[1] + 1/2
    x = np.exp(TIP * a)
    y = np.exp(TIP * b)
    z = 1 + 0j

    # our 9 phases are constrained by two equations, namely
    # [x[0], y[0]]^dag [x[3], y[3]] == 0
    # [x[1], y[1]]^dag [x[2], y[2]] == 0

    return create(x, y, z)


def create(x, y, z):
    q = np.stack([x, x, x, y, y, y]).T

    ws = np.ones((4, 6), dtype=np.complex128)
    abba = np.array([W, np.conjugate(W), np.conjugate(W), W])
    ws[:, 1] = abba
    ws[:, 4] = abba
    ws[:, 2] = np.conjugate(abba)
    ws[:, 5] = np.conjugate(abba)

    q *= ws

    p = np.array([z, z, z, -z, -z, -z])
    o = np.ones((6, ), dtype=np.complex128)

    result = np.concatenate([o[None, :], q, p[None, :]])
    return result


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


r = create_random()

# r = create(x, y, z)

np.set_printoptions(precision=5, suppress=True)

prod = np.conjugate(r.T).dot(r)
print(prod)
# unfortunately not true to 1e-6 atol.
assert np.allclose(np.abs(prod - 6 * np.eye(6)), 0, atol=1e-5)



def find_zrow(b):
    assert np.allclose(b[0, :], 1) # columns unphased. rows not.

    # which is the z, z, z, -z, -z, -z row?
    col = b[:, 0]
    zrow = None
    for i in range(1, 6):
        unphased = b[i, :] / col[i]
        angles = np.angle(unphased)
        rotated = np.mod(np.angle(unphased) + np.pi / 2, np.pi)
        if np.allclose(rotated, np.pi / 2, atol=1e-4):
            zrow = i
            break
    assert zrow is not None
    return zrow


def find_zrow_for_mub(a):
    zrows = set()
    for i in range(1, 4):
        zrows.add(find_zrow(a[i]))

    assert len(zrows) == 1, "the bases should be in sync"
    zrow = zrows.pop()
    return zrow


def find_triangles(b):
    row = b[1]
    x = row[0]
    mates = set()
    for i, y in enumerate(row):
        direction = (np.angle(x) - np.angle(y)) * 3 # this is supposed to be 0 mod 2pi if they belong together.
        if np.isclose(np.mod(direction + np.pi, TP), np.pi, atol=1e-4): # easier to check at the middle, no wrapover.
            mates.add(i)
    assert len(mates) == 3
    return mates


def verify(a, verbose=False):
    m, n = 4, 6
    for i in range(m):
        closeness = np.abs(6 * np.eye(6) - np.conjugate(a[i].T) @ a[i])
        assert np.allclose(closeness, 0, atol=1e-6)

    for i in range(m):
        for j in range(i + 1, m):
            aprod = np.abs(np.conjugate(a[i].T) @ a[j]) / 6
            distinct = tuple(np.unique((aprod * 1000).astype(int)))
            assert distinct == (355, 385, 425) or distinct == (408, )
            if verbose:
                print(i, j)
                print(aprod)

verify(a)
print("original verified to be close-MUB")


zrow = find_zrow_for_mub(a)


# permute rows in sync so that the +1, -1 row is the last:
a = a[:, list(range(zrow)) + list(range(zrow + 1, 6)) + [zrow], :]

zrow = find_zrow_for_mub(a)
assert zrow == 5

verify(a)
print("zrow to last did not ruin it")


for bi in range(1, 4):
    mates = find_triangles(a[bi])
    nonmates = set(range(6)) - mates
    perm = list(mates) + list(nonmates)
    print(perm)
    a[bi] = a[bi] @ np.eye(6)[:, perm]
    assert find_triangles(a[bi]) == set([0, 1, 2])

verify(a)
print("rearranging each basis into 3-3 columns did not ruin it")


def extract(b):
    x = b[1:5, 0]
    y = b[1:5, 3]
    z = b[5, 0]
    return x, y, z


def extract_angles(b):
    x, y, z = extract(b)
    return np.angle(x), np.angle(y), np.angle(z)


unphased = a.copy()
unphased[:, :, :3] /= unphased[:, :, 0:1]
unphased[:, :, 3:] /= unphased[:, :, 3:4]


ayes = []
bs = []
cs = []
graphs = []

for bi in range(1, 4):
    print(f"B_{bi}")
    aye, b, c = extract_angles(a[bi])
    print("a:", aye, "b:", b, "c:", z)
    ayes.append(aye)
    bs.append(b)
    cs.append(c)
    print("graph")
    angles = np.angle(unphased[bi])
    assert np.allclose(np.sin(angles * 3 / 2), 0, atol=1e-4)
    graph = np.round(angles / TP * 3).astype(int)
    assert np.all(graph[[0, 5], :] == 0)
    assert np.all(graph[:, [0, 3]] == 0)
    graphs.append(graph)

ayes = np.array(ayes) ; bs = np.array(bs) ; cs = np.array(cs) ; graphs = np.array(graphs)

print()
print("OR TO SUMMARIZE IT:")

np.set_printoptions(floatmode='unique')

print("ayes =", repr(ayes))
print("bs =", repr(bs))
print("cs =", repr(cs))
print("graphs =", repr(graphs))

np.set_printoptions(precision=5, suppress=True)


# undoing the scaling that we started with:
a /= 6 ** 0.5

if len(sys.argv) > 2:
    out_filename = sys.argv[2]
    np.save(out_filename, a)




xs = np.exp(1j * ayes)
ys = np.exp(1j * bs)
zs = np.exp(1j * cs)

Wnumeric = np.exp(TIP / 3)


def rebuild_single(x, y, z, graph):
    b = Wnumeric ** graph
    b[1:5, :3] *= x[:, None]
    b[1:5, 3:] *= y[:, None]
    b[5, :] *= z
    b[5, 3:] *= -1
    return b


def haagerup(H):
    hu = []
    for i in range(6):
        for j in range(6):
            for k in range(6):
                for l in range(6):
                    z = H[i,j]*H[k,l]*np.conjugate(H[i,l]*H[k,j])
                    hu.append(np.angle(z))
    return np.array(sorted(hu))


def discrete_haagerup(H, bins=360):
    hu = set()
    for i in range(6):
        for j in range(6):
            for k in range(6):
                for l in range(6):
                    z = H[i,j] * H[k,l] * np.conjugate(H[i,l] * H[k,j])
                    hu.add(np.around(np.angle(z) * bins / (2 * np.pi), 2))
    return np.array(sorted(hu))


def haagerup_distance(basis1, basis2):
    hu1 = haagerup(basis1)
    hu2 = haagerup(basis2)
    epsilon = 1e-7
    hu1f = hu1[np.abs(hu1) < np.pi - epsilon]
    hu2f = hu2[np.abs(hu2) < np.pi - epsilon]
    # a bit fragile, yes, but so far this assertion never failed, fingers crossed.
    assert len(hu1f) == len(hu2f)
    # this assumes that the angles are sorted,
    # and completely misbehaves when one of the angles goes from -pi+epsilon
    # to pi-epsilon, or vice versa.
    # that's why we threw away the ones closest in to pi in absolute value.
    return np.linalg.norm(hu1f - hu2f, ord=1)


def which_fourier(x, y, z, graph):
    return np.angle(x / y) * 180 / np.pi


def test_haagerup():
    x, y = np.exp(23j), np.exp(11j)
    v, u = np.exp(4j), np.exp(31j)
    epsilon = 1e-8
    b1 = canonical_fourier(x, y)
    b2 = canonical_fourier(v, u)
    b1b = canonical_fourier(y, x)
    b1c = canonical_fourier(-x, y)
    b1d = canonical_fourier(x, -y)
    assert haagerup_distance(b1, b2) > 10
    assert haagerup_distance(b1, b1b) < epsilon
    assert haagerup_distance(b1, b1c) < epsilon
    assert haagerup_distance(b1, b1d) < epsilon
    print("haagerup passed all verifications")


def degtophase(d):
    return np.exp(1j * d / 180 * np.pi)


def test_comparing_bases():
    x1d, y1d = 64.88793828463234, 119.9870819221207 # normalized/mub_10006.npy basis 1
    b1 = canonical_fourier(degtophase(x1d), degtophase(y1d))
    x2d, y2d = 59.987081922120666, 115.12497979324696 # normalized/mub_10006.npy basis 2
    b2 = canonical_fourier(degtophase(x2d), degtophase(y2d))
    x3d, y3d = 59.997177875439576, 64.88006781268074 # normalized/mub_1003.npy basis 1
    b3 = canonical_fourier(degtophase(x3d), degtophase(y3d))
    print("b1-b2", haagerup_distance(b1, b2))
    print("b1-b3", haagerup_distance(b1, b3))
    print("b2-b3", haagerup_distance(b2, b3))


def seriously_comparing_bases():
    bases = []
    for l in sys.stdin:
        indx, xd, yd = map(float, l.strip().split())
        indx = int(indx)
        basis = canonical_fourier(degtophase(xd), degtophase(yd))
        bases.append(basis)
    for i in range(len(bases)):
        for j in range(i, len(bases)):
            dist = haagerup_distance(bases[i], bases[j])
            print(i, j, dist)


# test_haagerup()
# test_comparing_bases() ; exit()
# seriously_comparing_bases() ; exit()


print("Okay, but which Fourier are they?")
for i in range(3):
    # print(which_fourier(ayes[i], bs[i], cs[i], graphs[i]))
    x = xs[i] ; y = ys[i] ; z = zs[i] ; graph = graphs[i]
    basis = rebuild_single(x, y, z, graph)
    prod = np.conjugate(basis.T) @ basis
    assert(np.allclose(prod, 6 * np.eye(6)))

    '''
    np.set_printoptions(precision=5, suppress=True, linewidth=100000)
    print(filename, i+1, len(discrete_haagerup(basis, bins=360)))
    continue
    '''

    candidate_parameters = x / y
    from itertools import combinations
    pairs = list(combinations(candidate_parameters, 2))
    for (p, q) in pairs:
        if not np.isclose(p, -q):
            break
    candidate_basis = canonical_fourier(p, q)
    dist = haagerup_distance(basis, candidate_basis)
    a = np.abs(np.angle(p)) / np.pi * 180
    b = np.abs(np.angle(q)) / np.pi * 180
    a, b = sorted([a, b])
    print("filename", filename, "basis", i + 1, "fourier_params_in_degrees", a, b, "haagerup_distance", dist)
