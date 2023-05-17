import numpy as np
import matplotlib.pyplot as plt
import itertools
import sys

from base import angler


# c is a cube. the piercing dimension is always the 0th axis.
def raw_h(c, gamma, axis=0, do_aggregate=True):
    assert c.shape == (6, 6, 6)
    assert gamma.shape == (6, )
    if axis == 0:
        gamma_e = gamma[:, None, None]
    elif axis == 1:
        gamma_e = gamma[None, :, None]
    elif axis == 2:
        gamma_e = gamma[None, None, :]

    powered = c ** gamma_e
    prods = np.prod(powered, axis=axis)
    if do_aggregate:
        return prods.sum()
    else:
        return prods

def conj_h(c, gamma, axis=0, do_aggregate=True):
    h = raw_h(c, gamma, axis=axis, do_aggregate=do_aggregate)
    return h + np.conjugate(h)


def perm_h(c, gamma, axis=0, do_conj=True, do_aggregate=True):
    s = 0
    n = 0
    for gamma_permuted in itertools.permutations(gamma):
        gamma_permuted = np.array(gamma_permuted)
        if do_conj:
            s += conj_h(c, gamma_permuted, axis=axis, do_aggregate=do_aggregate)
        else:
            s += raw_h(c, gamma_permuted, axis=axis, do_aggregate=do_aggregate)
        n += 1
    s /= n
    return s


filename, = sys.argv[1:]
c = np.load(filename)
assert c.shape == (6, 6, 6), "please provide a Hadamard cube"

print("normalizing to unimodular, to avoid floating point issues when multiplying")
c /= np.abs(c)


n = 6
N = 4

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

do_axis_summing = False
prioritized_axis = 0

gamma = np.ones(6, dtype=np.complex128)
gamma[0] += 1
gamma[-1] -= 1

# print(perm_h(c, gamma, axis=prioritized_axis, do_aggregate=True))


gamma_0 = np.zeros(6, dtype=np.complex128)
assert not do_axis_summing, "h(0) comparison not yet implemented with axis summing"
zero_value = perm_h(c, gamma_0, axis=prioritized_axis, do_conj=True)
print("at all-0:", zero_value)

for gamma in b:
    if do_axis_summing:
        h = 0
        for axis in range(3):
            h_directional = perm_h(c, gamma, axis=axis, do_conj=True)
            h += h_directional
    else:
        h = perm_h(c, gamma, axis=prioritized_axis, do_conj=True)
    assert np.isclose(h.imag, 0)

    if np.isclose(h.real, 0):
        # print(f"{gamma}, {h.real:.6f}")
        print(gamma, h.real)
