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


def create_random():
    x = np.exp(TIP * np.random.uniform(size=4))
    y = np.exp(TIP * np.random.uniform(size=4))
    z = np.exp(TIP * np.random.uniform())

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

r = create(np.squeeze(x), np.squeeze(y), z)

np.set_printoptions(precision=5, suppress=True)

prod = np.conjugate(r.T).dot(r)
print(prod)
# unfortunately not true to 1e-6 atol.
assert np.allclose(np.abs(prod - 6 * np.eye(6)), 0, atol=1e-5)





