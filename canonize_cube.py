import sys
import numpy as np
from itertools import permutations
import matplotlib.pyplot as plt

from sympy import *
from sympy.physics.quantum.dagger import Dagger
from sympy.tensor.array import MutableDenseNDimArray

from base import *



# assumes integer input
def check_circulant(s):
    return s[0,0]==s[1,1]==s[2,2] and s[1,0]==s[2,1]==s[0,2] and s[0,1]==s[1,2]==s[2,0]


# assumes integer input
def find_block_circulant_structure(b):
    perms = list(permutations(range(6)))

    for p1 in perms:
        for p2 in perms:
            bb = b[p1, :]
            bb = bb[:, p2]
            if check_circulant(bb[:3, :3]) and check_circulant(bb[3:, :3]) and check_circulant(bb[:3, 3:]) and check_circulant(bb[3:, 3:]):
                return p1, p2
    return None


def test_find_block_circulant_structure(c):
    b = c[:, :, 0]
    b = np.round(angler(b)).astype(int)
    s = find_block_circulant_structure(b)
    if s is not None:
        p1, p2 = s
        bb = b[p1, :]
        bb = bb[:, p2]
        print(bb)
    else:
        print("not block circulant")


# test_find_block_circulant_structure(c) ; exit()


def plot_parquet_zeros(c):
    # c is transposed at the end of the for block!
    for i in range(3):
        print("direction", i)

        can = get_canonizer(c[0, :, :])
        print("F(", angler(can['x']), ",", angler(can['x']))

        for k in range(-7, 8):
            for l in range(-7, 8):
                cc = c * 6 ** 0.5
                cc[:, 3:, :] **= k
                cc[:, :3, :] **= l
                if np.linalg.norm(cc.prod(axis=1).sum(axis=0)) < 1e-8:
                    print("X", end=" ")
                else:
                    print("O", end=" ")
            print()
        c = c.transpose([1, 2, 0])


# plot_parquet_zeros(c) ; exit()


# a[indx]^dag is left-applied to everybody. so that it turns into Id.
# then MUB elements are rotated among themselves
# so that a[index] becomes a[0]
def swap_identity(a, indx):
    if indx == 0:
        return a.copy()

    assert np.allclose(a[0], np.eye(6))
    a2 = a.copy()
    a2[1] = trans(a[indx], a[0])
    a2[2] = trans(a[indx], a[3 - indx])
    verify_mub(a2)
    return a2


def standardize_triplet_order(a):
    a_orig = a.copy()
    for i in range(3):
        a = swap_identity(a_orig, i)
        result1 = find_blocks(a[1], allow_transposal=False)
        result2 = find_blocks(a[2], allow_transposal=False)
        if result1 is not None and result2 is not None:
            # not transposed
            assert not result1[2] and not result2[2]
            return a
    return None


# a canonical hadamard has its bipartition of columns interlaced like 010101
# and its tripartition of rows interlaced like 012012.
# here we rearrange it into 000111, 001122.
def deinterlace(b_orig):
    b = b_orig[[0, 3, 1, 4, 2, 5], :]
    b = b[:, [0, 2, 4, 1, 3, 5]]
    return b


def hardwired_reorder(a_orig):
    a = a_orig.copy()
    a[1] = deinterlace(a[1])
    a[2] = deinterlace(a[2])
    return a


def mub_type(a):
    # by convention
    # counts[0] is number of Fourier,
    # counts[1] is number of Fourier-transposed
    # counts[2] is number of neither.
    counts = [0, 0, 0]
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            result = find_blocks(trans(a[i], a[j]), allow_transposal=True)
            if result is None:
                counts[2] += 1
            else:
                _, _, is_transposed = result
                counts[int(is_transposed)] += 1
    if counts == [6, 0, 0]:
        return "sporadic"
    elif counts == [2, 2, 2]:
        return "Szollosi"
    else:
        return "WTF"


def truncate_canon(g, also_left_perm=False):
    Id = np.eye(6, dtype=np.complex128)
    # The magnitude of gi['d_r'] is 1/sqrt(6), we have to put that
    # back if we drop gi['d_r']
    if also_left_perm:
        phases, perm_m = np.diag(g['d_l']), g['p_l']
        switched_perm_m, switched_phases = switch_phase_perm(phases, perm_m)
        return create_generalized_fourier(np.diag(switched_phases), Id, g['x'], g['y'], Id, Id / 6 ** 0.5)
    else:
        create_generalized_fourier(g['d_l'], g['p_l'], g['x'], g['y'], Id, Id / 6 ** 0.5)


def is_row_structure_compatible(b1, b2):
    result_1 = find_blocks(b1, allow_transposal=False)
    bipart_col_1, tripart_col_1, is_transposed_1 = result_1
    result_2 = find_blocks(b2, allow_transposal=False)
    bipart_col_2, tripart_col_2, is_transposed_2 = result_2
    return tripart_col_1 == tripart_col_2


# more percisely, find_permutation_that_turns_blocks_to_circulant()
# apply at most 2 transpositons to a[2] columns
# so that sx becomes block-circulant.
# permuting a[2] columns permutes sx columns,
# permuting a[1] columns permutes sx rows.
# assumes that the bipartition is standard (012)(345).
def turn_blocks_to_circulant(b):
    perm = list(range(6))
    w = visualize_clusters(b, group_conjugates=False)
    if w[2, 1] == w[1, 2]:
        perm[2], perm[1] = perm[1], perm[2]
    if w[4, 5] == w[5, 4]:
        perm[4], perm[5] = perm[5], perm[4]
    return perm


def check_phi_property(b, atol=1e-4):
    b2 = szollosi_original(b[0, :])
    ratio = b / b2
    phi_candidate = ratio[3, 0]
    ratio[3:, :] /= phi_candidate
    if np.allclose(ratio, 1, atol=atol):
        return phi_candidate
    else:
        return None


def turn_lower_half_to_single_phi(a_orig):
    a = a_orig.copy()
    for i in range(3):
        c = hadamard_cube(a)
        phi = check_phi_property(c[0, :, :])
        if phi is not None:
            return a, phi
        a[1] = a[1][:, [0, 1, 2, 4, 5, 3]]
        # print("rotating second triangle of a[1] to get constant phi")
    assert False, "none of the rotations leads to a constant phi"


# 1. rearranges the MUB columns and MUB rows so that
#    the cube has a block structure consisting of 3x3x2 blocks.
# 2. permutes within block so that each 3x3 is circulant.
# 3. permutes the lower half of the sx slices
#    (more exactly, the a[1] columns) so that the lower half
#    is constant phi times the Szollosi formula (6).
def reorder_mub(a_orig):
    a = a_orig.copy()
    assert is_row_structure_compatible(a[1], a[2])
    # if yes, we can safely apply it individually to permute rows.
    a[1] = arrange_blocks(a[1])
    a[2] = arrange_blocks(a[2])
    verify_mub(a)
    c = hadamard_cube(a)
    verify_cube_properties(c)

    perm = turn_blocks_to_circulant(c[0, :, :])
    a[2] = a[2][:, perm]

    c = hadamard_cube(a)
    verify_cube_properties(c)

    perm = turn_blocks_to_circulant(c[0, :, :])
    assert perm == list(range(6))

    a, phi = turn_lower_half_to_single_phi(a)

    return a


# does two things:
# 1. deduces their generalized F-form from the MUB elements,
#    and removes the superfluous right actions.
# 2. permutes the MUB rows and columns so that
#    the appropriate block-circulant structure and conjugate
#    pairing appears.
def standardize_mub(a):
    b1_canon = get_canonizer(a[1])
    b2_canon = get_canonizer(a[2])
    # assert np.all(b1_canon['p_l'] == b2_canon['p_l'])

    b1_truncated_canon = truncate_canon(b1_canon, also_left_perm=True)
    b2_truncated_canon = truncate_canon(b2_canon, also_left_perm=True)
    dl1 = b1_truncated_canon['d_l']
    b2_truncated_canon['d_l'] *= np.conjugate(dl1)
    b1_truncated_canon['d_l'] *= np.conjugate(dl1)

    def pretty(canon):
        print(angler(np.diag(canon['d_l'])), angler(canon['x']), angler(canon['y']))
    # pretty(b1_truncated_canon)
    # pretty(b2_truncated_canon)

    a[1] = rebuild_from_canon(b1_truncated_canon)
    a[2] = rebuild_from_canon(b2_truncated_canon)

    verify_mub(a)
    c = hadamard_cube(a)
    verify_cube_properties(c)

    # before all the left perm removal and such,
    # reorder_mub(a) was needed to bring the MUB to standard form.
    # after it, hardwired_reorder(a) is enough, a fixed, trivial rearrange.
    # a = reorder_mub(a)
    a = hardwired_reorder(a)

    verify_mub(a)
    c = hadamard_cube(a)
    verify_cube_properties(c)

    x1y1 = (b1_truncated_canon['x'], b1_truncated_canon['y'])
    dl2 = b2_truncated_canon['d_l']
    x2y2 = (b2_truncated_canon['x'], b2_truncated_canon['y'])
    return a, x1y1, dl2, x2y2


def save(basefilename, arr, ext):
    filename = f"{basefilename}.{ext}"
    if ext == "npy":
        np.save(filename, arr)
    elif ext == "mat":
        from scipy.io import savemat
        savemat(filename, {"data": arr})
    else:
        assert False, "unknown extension"


def main():
    np.set_printoptions(precision=5, suppress=True, linewidth=160)

    code, ext = sys.argv[1:]
    filename = f"triplets/triplet_mub_{code}.npy"
    a = np.load(filename)

    if len(a) == 2:
        a = np.stack([np.eye(6, dtype=np.complex128), a[0], a[1]])

    a_original = a

    verify_mub(a)
    c = hadamard_cube(a)
    verify_cube_properties(c)

    save(f"straight_cubes/cube_{code}", hadamard_cube(a_original), ext)

    save(f"straight_triplets/triplet_mub_{code}", a_original, ext)

    mub_type_name = mub_type(a)
    if mub_type_name != "Szollosi":
        print("filename", filename, mub_type_name)
        exit()

    a = standardize_triplet_order(a)
    a, x1y1, dl2, x2y2 = standardize_mub(a)

    c = hadamard_cube(a)
    verify_cube_properties(c)

    save(f"canonized_cubes/canonized_cube_{code}", c, ext)

    a_prime = cube_to_mub(c)
    # a_prime = cube_to_mub_simplified(c)
    verify_mub(a_prime)



if __name__ == "__main__":
    main()
