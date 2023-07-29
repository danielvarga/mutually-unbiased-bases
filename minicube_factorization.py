import sys
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


filename, = sys.argv[1:] # "canonized_cubes/canonized_cube_00018.npy"
c = np.load(filename)

'''
# TODO something here
yperm = [0, 5, 1, 4, 2, 3]
zperm = [0, 4, 1, 3, 2, 5] # [0, 1, 2, 5, 4, 3]

c = c[:, yperm, :]
c = c[:, :, zperm]

print(f"applying yperm {yperm} zperm {zperm}. managed to make it out of proper 2x2x2 blocks?:", proper_2x2x2_blocks(c))
'''

# not just any canonizer, but the lexicographically smallest one
def find_canonizer(c):
    ps = list(itertools.permutations(range(6)))
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
        return yperm, zperm
    else:
        return None


def find_all_canonizers(c):
    ps = list(itertools.permutations(range(6)))
    c_orig = c.copy()
    found = []
    for yperm in ps:
        for zperm in ps:
            c_permuted = c_orig[:, yperm, :]
            c_permuted = c_permuted[:, :, zperm]
            if proper_2x2x2_blocks(c_permuted):
                found.append((yperm, zperm))
    return sorted(found)


def code_to_perm(code):
    p = np.arange(6, dtype=int)
    if code % 2 == 1:
        p = p[[1, 0, 2, 3, 4, 5]]
    if (code // 2) % 2 == 1:
        p = p[[0, 1, 3, 2, 4, 5]]
    if (code // 4) % 2 == 1:
        p = p[[0, 1, 2, 3, 5, 4]]
    return p


def deinterlace(c):
    c = c[:, [0, 2, 4, 1, 3, 5], :]
    c = c[:, :, [0, 2, 4, 1, 3, 5]]
    return c


def mirror(c):
    c = c[:, [0, 1, 2, 5, 4, 3], :]
    c = c[:, :, [0, 1, 2, 5, 4, 3]]
    return c


def deinterlace_and_mirror(c):
    return mirror(deinterlace(c))


# a tricanonizer permutes the second two axes of a canonized cube so that it's tricanonized while remaining canonized.
# canonized here means:
#    - first axis is Szollosi,
#    - first axis has 3x2 block structure (aka dual pairs)
#    - whole cube has a 3x3x3 block structure of 2x2x2 blocks,
#             where each 2x2x2 is a minicube aka a cube with all slices being Hadamard.
# tricanonized means:
#    after applying the fixed deinterlacing permutations [:, [0, 2, 4, 1, 3, 5], :] and [:, :, [0, 2, 4, 1, 3, 5]],
#    and also applying the fixed mirroring permutations [:, [0, 1, 2, 5, 4, 3], :] and [:, :, [0, 1, 2, 5, 4, 3]],
#    the Szollosi slices are in the standard block-circulant (although bistochastic) form.
# out of the (2^3)^2 potential tricanonizers (perm_y, perm_z), exactly 2 is an actual tricanonizer,
# and you get one from the other by transposing all three pairs, as in
# [0 1 2 3 4 5] [0 1 2 3 4 5]
# [1 0 3 2 5 4] [1 0 3 2 5 4]
# UPDATE: it seems like tricanonization is a no-op, [0 1 2 3 4 5] [0 1 2 3 4 5] is always a tricanonizer, because earlier the canonizer 
# has found the lexicographically smallest canonization starting from a block-circulant MUB-cube.
def find_tricanonizer(c_orig):
    for code_y in range(8):
        perm_y = code_to_perm(code_y)
        for code_z in range(8):
            perm_z = code_to_perm(code_z)
            c = c_orig.copy()
            c = c[:, perm_y, :]
            c = c[:, :, perm_z]
            c_triblock = deinterlace_and_mirror(c)
            diag1 = np.diag(c_triblock[0, :3, :3])
            diag2 = np.diag(c_triblock[0, 3:, 3:])
            if np.allclose(diag1, diag1.mean()) and np.allclose(diag2, diag2.mean()):
                print(perm_y, perm_z)
                # return perm_y, perm_z
    raise Exception("no tricanonizer found")


'''
found = find_all_canonizers(c)
print("number of 2x2x2 canonizers", len(found)) ; exit()
'''

result = find_canonizer(c)
if result is None:
    print("filename", filename, "no canonizer found")
    exit()
else:
    yperm, zperm = result
    c = c[:, yperm, :]
    c = c[:, :, zperm]
    assert proper_2x2x2_blocks(c)

    try:
        yperm_tri, zperm_tri = find_tricanonizer(c)
    except:
        print("filename", filename, "no tricanonizer found")
        exit()

    c = c[:, yperm_tri, :]
    c = c[:, :, zperm_tri]
    assert proper_2x2x2_blocks(c)

    cv = visualize_clusters(c, group_conjugates=False) ; print(cv)
