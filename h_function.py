import numpy as np
import matplotlib.pyplot as plt
import itertools
import sys


# c is a cube. the piercing dimension is always the 0th axis.
def raw_h(c, gamma, axis=0):
    assert c.shape == (6, 6, 6)
    assert gamma.shape == (6, )
    powered = c ** gamma[:, None, None]
    prods = np.prod(powered, axis=axis)
    return prods.sum()

def conv_h(c, gamma, axis=0):
    h = raw_h(c, gamma, axis=axis)
    return h + np.conjugate(h)


def perm_h(c, gamma, axis=0, do_conv=True):
    s = 0
    n = 0
    for gamma_permuted in itertools.permutations(gamma):
        gamma_permuted = np.array(gamma_permuted)
        if do_conv:
            s += conv_h(c, gamma_permuted, axis=axis)
        else:
            s += raw_h(c, gamma_permuted, axis=axis)
        n += 1
    s /= n
    return s


filename, = sys.argv[1:]
c = np.load(filename)
assert c.shape == (6, 6, 6), "please provide a Hadamard cube"

c /= np.abs(c)


n = 6
N = 5

x = np.arange(-N+1, N, dtype=int)
xs = [x] * n
# nth direct power of [-N, N] interval:
a = np.meshgrid(*xs)
# array of all possible [-N, N] intervals:
b = np.stack(a, axis=-1).reshape((-1, n))

for i, gamma in enumerate(b):
    if sum(gamma) < 0:
        b[i] = - b[i]

# each re-ordered descending:
b = - np.sort(- b, axis=-1)
# keeping only the distinct ones:
b = np.unique(b, axis=0)

for gamma in b:
    '''
    h = 0
    for axis in range(3):
        h_directional = perm_h(c, gamma, axis=axis, do_conv=True)
        h += h_directional
    '''
    h = perm_h(c, gamma, axis=0, do_conv=True)
    assert np.isclose(h.imag, 0)
    if np.isclose(h.real, 0):
        # print(f"{gamma}, {h.real:.6f}")
        print(gamma, h.real)
