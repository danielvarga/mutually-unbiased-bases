# python fortyeight_extract_tao.py | sed "s/, $//" | sed "s/ 0\+d/ d/g" | sed "s/ 0-delta/ -delta/g"
# delta = 44.477512185929605
# 0, delta, 0, 120, 120, -120+delta
# 0, 120+delta, 0, delta, delta, 120+delta
# 0, 0, delta, -120+delta, 120, 120
# 0, 0, -120, -120, delta, delta
# ...etcetera, 90 rows

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
F = tao()

m = np.load("fortyeight_tao.npy")
print(m.shape)
m /= m[:, :1]

# throw away the ones that haven't converged:
appropriate_length = np.isclose(np.linalg.norm(m, axis=1), 6 ** 0.5)
m = m[appropriate_length]
print(m.shape, "left after filtering for vector norm")

appropriate_elements = np.isclose(np.abs(m), 1)
appropriate_elements = appropriate_elements.min(axis=1)
print("appropriate_elements", appropriate_elements.shape)
m = m[appropriate_elements]
print(m.shape, "left after filtering phase elements")

appropriate_bias = np.isclose(np.abs(np.conjugate(F) @ m.T), 6 ** 0.5)
assert appropriate_bias.shape == (6, len(m))
appropriate_bias = appropriate_bias.min(axis=0)
m = m[appropriate_bias]
print(m.shape, "left after filtering unbiasedness to F")



alldist = np.abs(m[:, :, None, None] - m[None, None, :, :])
assert alldist.shape == (len(m), 6, len(m), 6)
dist = np.diagonal(alldist, axis1=1, axis2=3)
assert dist.shape == (len(m), len(m), 6)
dist = dist.sum(axis=2)
match = dist < 1e-5
match_unique, indices = np.unique(match, axis=1, return_index=True)
print(indices.shape)
assert indices.shape == (90, ) # this seems to be true for Tao

n = len(indices)

uniques = m[indices]

np.set_printoptions(precision=9, suppress=True)


np.save("fortyeight_tao_really.npy", uniques)

angles = angler(uniques)

# this is a particular, non-generalizing way of getting 0.5222:
delta = angles[0, 1]
print("delta =", delta)
atol = 1e-4
integers = np.isclose(angles, np.round(angles), atol=atol)
left_rotateds = np.isclose(angles - delta, np.round(angles - delta), atol=atol)
right_rotateds = np.isclose(angles + delta, np.round(angles + delta), atol=atol)
# they cover all the cases:
assert np.allclose(integers.astype(int) + left_rotateds.astype(int) + right_rotateds.astype(int), np.ones_like(integers.astype(int)))

# TODO that's ugly
def standardize(integers_row, left_rotateds_row, right_rotateds_row, angles_row):
    angles = []
    delta_coeffs = []
    for i in range(6):
        if integers_row[i] == 1:
            angles.append(np.round(angles_row[i])) ; delta_coeffs.append(0)
        elif left_rotateds_row[i] == 1:
            angles.append(np.round(angles_row[i] - delta)) ; delta_coeffs.append(1)
        elif right_rotateds_row[i] == 1:
            angles.append(np.round(angles_row[i] + delta)) ; delta_coeffs.append(-1)
        else:
            assert False
    return angles, delta_coeffs


def pretty(angles, delta_coeffs):
    pretty_dict = {-1: "-delta", 0: "", 1: "+delta"}
    for i in range(len(angles)):
        angle = angles[i]
        assert np.isclose(angle / 15, np.round(angle / 15)), f"bad angle {angle}"
        delta_coeff = delta_coeffs[i]
        assert delta_coeff in (-1, 0, 1)
        pretty = str(int(np.round(angles[i]))) + pretty_dict[delta_coeff]
        pretty = pretty.strip()
        print(pretty, end=", ")
    print()

for integers_row, left_rotateds_row, right_rotateds_row, angles_row in list(zip(integers.astype(int), left_rotateds.astype(int), right_rotateds.astype(int), angles)):
    angles, delta_coeffs = standardize(integers_row, left_rotateds_row, right_rotateds_row, angles_row)
    pretty(angles, delta_coeffs)


exit()




pairwise = uniques @ np.conjugate(uniques.T)
print(pairwise.shape)
assert pairwise.shape == (90, 90)
orthogonal = np.isclose(pairwise, 0)
print(orthogonal, orthogonal.sum())
exit()

combs = len(list(combinations(range(n), 6)))
print(combs)
mats = []
for i, row_indices in enumerate(combinations(range(n), 6)):
    mat = uniques[list(row_indices)]
    if np.allclose(trans(mat, mat), np.eye(6) * 6, atol=1e-4):
        print(">", row_indices)
        mats.append(mat)

    if i % 100000 == 0:
        print(i)

mats = np.array(mats)
print(mats.shape)
np.save("fortyeight_tao_structured.npy", mats)
