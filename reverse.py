# the goal is to re-create mub_120_normal.npy B_1 as parametrized matrix

import sys
import numpy as np


TP = 2 * np.pi
TIP = 2j * np.pi
W = np.exp(TIP / 3)


# mub_120_normal.npy
filename, = sys.argv[1:]
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



# this does not have any apparent structure:
'''
import matplotlib.pyplot as plt
for bi in range(1, 4):
    x, y, z  = extract_angles(a[bi])
    plt.scatter(x * 360 / TP, y * 360 / TP)

plt.show()
'''
