import sys
import numpy as np
from itertools import permutations
import matplotlib.pyplot as plt

import sympy

from base import *


np.set_printoptions(precision=5, suppress=True, linewidth=160)


# triplets/triplet_mub_00018.npy generic
# triplets/triplet_mub_00045.npy sporadic
filename = sys.argv[1]
a = np.load(filename)

if len(a) == 2:
    a = np.stack([np.eye(6, dtype=np.complex128), a[0], a[1]])


verify_mub(a)
c = hadamard_cube(a)
verify_cube_properties(c)


def find_uniques_original(c, group_conjugates=False):
    n = 6
    N = 10000
    bins = (N * np.angle(c)).astype(int)
    if group_conjugates:
        bins = np.abs(bins)
    vals, indx, cnts = np.unique(bins.flatten(), return_counts=True, return_index=True)
    uniques = c.flatten()[indx]
    return uniques

def find_uniques(c, group_conjugates=False):
    n = 6
    N = 10000
    bins = (N * np.angle(c)).astype(int)
    if group_conjugates:
        bins = np.abs(bins)
    vals, indx, cnts = np.unique(bins.flatten(), return_counts=True, return_index=True)
    print(cnts)
    uniques = c.flatten()[indx]
    uqs = []
    if set(cnts) == set((3, )):
        print("everything appears three times")
        uqs = uniques
    else:
        for u, cnt in zip(uniques, cnts):
            uqs += [u] * (cnt // 3)

    return np.array(uqs)



us = find_uniques(c, group_conjugates=False)

plt.scatter(us.real, us.imag)
plt.show()

x = sympy.symbols('x')
product = None
for u in us:
    if product is None:
        product = (x - u)
    else:
        product *= (x - u)

poly = sympy.expand(product).as_poly(domain="C")
coeffs = poly.all_coeffs()
coeffs = np.array(coeffs).astype(np.complex128)

print(coeffs)
print(len(coeffs))