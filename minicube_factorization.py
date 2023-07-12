import numpy as np
import itertools

from base import *


def is_hadamard(m):
    assert m.shape == (2, 2)
    return np.isclose(m[0, 0] * m[1, 1] + m[0, 1] * m[1, 0], 0)


def is_hadamard_minicube(minicube):
    for axis in (0, 1, 2):
        for coord in (0, 1):
            slc = slic(minicube, axis, coord)
            if not is_hadamard(slc):
                return False
    return True


def find_minicubes(c):
    pairs = list(itertools.combinations(range(6), 2))
    for xp in pairs:
        for yp in pairs:
            for zp in pairs:
                m = c.copy()
                m = m[xp, :, :]
                m = m[:, yp, :]
                m = m[:, :, zp]
                if is_hadamard(m[0, :, :]):
                    if is_hadamard_minicube(m):
                        print(xp, yp, zp)


def proper_2x2x2_blocks(c):
    for i in range(0, 6, 2):
        for j in range(0, 6, 2):
            for k in range(0, 6, 2):
                minicube = c[i:i+2, j:j+2, k:k+2]
                if not is_hadamard_minicube(minicube):
                    return False
    return True



c = np.load("canonized_cubes/canonized_cube_00018.npy")

# TODO something here
yperm = [0, 3, 1, 5, 2, 4]
zperm = [0, 5, 1, 4, 2, 3]

c = c[:, yperm, :]
c = c[:, :, zperm]



ps = list(permutations(range(6)))
c_orig = c.copy()
found = False
for yperm in ps:
    for zperm in ps:
        c_permuted = c_orig[:, yperm, :]
        c_permuted = c_permuted[:, :, zperm]
        if proper_2x2x2_blocks(c_permuted):
            found = True
            break
    if found:
        break
if found:
    print("FOUND IT!")
    print("yperm", yperm, "zperm", zperm)
    cv = visualize_clusters(c, group_conjugates=False)
    print(cv)
else:
    print("nope :(")
