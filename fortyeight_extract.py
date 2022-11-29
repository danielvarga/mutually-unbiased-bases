from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt

from base import *

# this was created from
np.random.seed(1)
def random_phases(shape):
    return np.exp(2 * np.pi * 1j * np.random.uniform(size=shape))
x, y = random_phases((2, ))
F = canonical_fourier(x, y)
print(x, y)

m = np.load("fortyeight.npy")
print(m.shape)
m /= m[:, :1]

alldist = np.abs(m[:, :, None, None] - m[None, None, :, :])
assert alldist.shape == (len(m), 6, len(m), 6)
dist = np.diagonal(alldist, axis1=1, axis2=3)
assert dist.shape == (len(m), len(m), 6)
dist = dist.sum(axis=2)
match = dist < 1e-10
match_unique, indices = np.unique(match, axis=1, return_index=True)
assert indices.shape == (48, ) # Jaming et al 1.3.

uniques = m[indices]

np.save("fortyeight_really.npy", uniques)

combs = len(list(combinations(range(48), 6)))
print(combs)
mats = []
for i, row_indices in enumerate(combinations(range(48), 6)):
    mat = uniques[list(row_indices)]
    if np.allclose(trans(mat, mat), np.eye(6) * 6, atol=1e-4):
        print(">", row_indices)
        mats.append(mat)

    if i % 100000 == 0:
        print(i)

mats = np.array(mats)
print(mats.shape)
np.save("fortyeight_structured.npy", mats)
