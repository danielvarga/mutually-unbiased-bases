import itertools
import numpy as np


def standalone_formula(c, gamma):
    n = c.shape[0]
    assert c.shape == (n, n, n)
    assert gamma.shape == (n, )
    # first we do the powering on axis=1.
    # then we multiply over axis=1.
    # then we sum over axis=2.
    # then we take magnitude squared.
    # then we average over axis=0.
    powered = c ** gamma[None, :, None]
    vs = powered.prod(axis=1).sum(axis=1)
    values = np.abs(vs) ** 2
    values /= n ** 2
    return values.mean()


cubes = np.load("all_straight_cubes.npy")
c_orig = cubes[5]
c_orig *= 6 ** 0.5

gamma = np.array([1, 1, 1, 0, 0, -2])
gamma = np.array([3, 0, 0, 0, 0, -2])


for axis_perm in itertools.permutations(range(3)):
    c = np.transpose(c_orig, axis_perm)
    gg = standalone_formula(c, gamma)
    print(axis_perm, gg)
