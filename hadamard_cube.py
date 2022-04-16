import sys
import numpy as np
from itertools import permutations
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D

from base import *


def visualize_clusters(c, group_conjugates=True):
    n = 6
    N = 10000
    bins = (N * np.angle(c)).astype(int)
    if group_conjugates:
        bins = np.abs(bins)
    vals, cnts = np.unique(bins.flatten(), return_counts=True)

    # for a full cube every color appears 6 times.
    #   (or 3 times if conjugates are not grouped)
    # for a single slice it's either all distinct or
    # every color appears 3 times.
    # for sporadic Fouriers (6th roots of unity) it's 12 or 6 per color.
    assert np.all(cnts == 6) or np.all(cnts == 1) or np.all(cnts == 3) \
        or sorted(cnts) == [6, 6, 12, 12]
    dists = bins[..., None] - vals[None, :]
    close = np.isclose(dists, 0)
    which = np.argmax(close, axis=-1)
    return which


def visualize(c):
    cmap = plt.cm.viridis

    n = 6
    if True:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        patches = []
        colors = []
        for row in range(n):
            for col in range(n):
                centre = 3 * row + 3j * col
                z = centre
                scalprod = 0
                for k in range(n):
                    use_segments = False
                    if not use_segments:
                        z = centre
                    delta = c[row, col, k]
                    ax.arrow(np.real(z), np.imag(z), np.real(delta), np.imag(delta), head_width=0.02, length_includes_head=True, color=plt.get_cmap('Dark2')(k))
                    z += delta
                    scalprod += delta
                z -= centre
                radius = np.abs(scalprod)
                circle = Circle((centre.real, centre.imag), radius)
                patches.append(circle)
                colors.append(1)
        p = PatchCollection(patches, alpha=0.05, cmap=cmap)
        p.set_array(np.array(colors))
        ax.add_collection(p)

        plt.title("Hadamard cube")
        plt.xlim((-3, 18))
        plt.ylim((-3, 18))
        plt.savefig("cube.pdf")
        # plt.show()


def visualize_3d(c):
    from mpl_toolkits.mplot3d import axes3d
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    x, y, z = np.meshgrid(np.arange(6), np.arange(6), np.arange(6))
    w = np.zeros_like(x)
    ax.quiver(x, y, z, c.real, c.imag, w, length=0.5)
    plt.show()


if len(sys.argv[1:]) > 1:
    filenames = sys.argv[1:]
    cs = []
    kept = []
    for filename in filenames:
        a = np.load(filename)
        c = hadamard_cube(a)
        try:
            verify_cube_properties(c)
            cs.append(c)
            kept.append(filename)
        except:
            pass
    size_rounded = int(len(cs) ** 0.5)
    cs = cs[:size_rounded * size_rounded]
    fig, axes = plt.subplots(size_rounded, size_rounded, figsize=(20, 20))
    axes = [ax for axline in axes for ax in axline]
    sq6 = 1 / 6 ** 0.5 + 0.02
    for ax, c, filename in zip(axes, cs, kept):
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.set_xlim((-sq6, sq6))
        ax.set_ylim((-sq6, sq6))
        ax.set_aspect('equal')
        ax.set_title(filename)
        ax.scatter(c.flatten().real, c.flatten().imag, s=0.5)
    plt.savefig("many_triplets_b.pdf")
    exit()


# dead code, because there are NO nice angles.
def nice_angles(c):
    div = c[:, None] / c[None, :]
    n = 24
    dn = div ** n
    # these are all simply repeat elements, first roots of unity:
    print("nice ratios between Hadamard cube elements", np.isclose(dn, 1, atol=1e-4).astype(int).sum())




# np.set_printoptions(precision=12, suppress=True, linewidth=100000)
np.set_printoptions(precision=5, suppress=True)

# triplet_mub_015.npy
filename = sys.argv[1]
a = np.load(filename)

a = np.stack([np.eye(6, dtype=np.complex128), a[0], a[1]])

for i in range(1, len(a)):
    verify_hadamard(a[i])

verify_mub(a)

c = hadamard_cube(a)

verify_cube_properties(c)



def counts(c):
    N = 100000
    bins = (N * np.angle(c)).astype(int)
    bins = np.abs(bins)
    vals, cnts = np.unique(bins.flatten(), return_counts=True)
    return cnts


def verify_surprising_cube_properties(c):
    np.set_printoptions(precision=5, suppress=True, linewidth=100000)
    cnts = counts(c)
    cnts = -np.sort(-cnts)
    print(cnts, end="\t")
    if cnts[0] == 72:
        assert sorted(cnts) == [36, 36, 72, 72]
        print("sporadic")
        return
    found = False
    for direction in range(3):
        s = slic(c, direction, 0)
        cnts = counts(s)
        if np.min(cnts) >= 3:
            found = True
            break
    assert found
    is_transposeds = []
    for d in set(range(3)) - set([direction]):
        s = slic(c, d, 0)
        bipart_col, tripart_row, is_transposed = find_blocks(s, allow_transposal=True)
        assert len(bipart_col) == 2 and len(bipart_col[0]) == 3 and len(bipart_col[1]) == 3
        assert len(tripart_row) == 3 and len(tripart_row[0]) == 2 and len(tripart_row[1]) == 2 and len(tripart_row[2]) == 2
        is_transposeds.append(is_transposed)
    # assert is_transposeds[0] == is_transposeds[1]
    print("Szollosi", " ".join("F" if not is_transposed else "FT" for is_transposed in is_transposeds))


verify_surprising_cube_properties(c) ; exit()


if filename == "triplets/triplet_mub_00018.npy":
    x_perm = [0, 1, 2, 3, 4, 5]
    y_perm = [0, 4, 1, 2, 3, 5]
    z_perm = [0, 4, 2, 3, 5, 1]
elif filename == "triplets/triplet_mub_00171.npy":
    x_perm = [0, 2, 1, 5, 3, 4]
    y_perm = [0, 1, 2, 3, 4, 5]
    z_perm = [0, 5, 1, 2, 4, 3]
else:
    print("WARNING: no manually computed permutations available")
    x_perm = y_perm = z_perm = range(6)
c = c[x_perm, :, :]
c = c[:, y_perm, :]
c = c[:, :, z_perm]

verify_cube_properties(c)


def test_cube_invariances(a_orig):
    a = a_orig.copy()
    verify_mub(a)
    c1 = hadamard_cube(a)
    col_perm = [1, 0, 3, 4, 2, 5]
    col_perm_m = np.eye(6)[:, col_perm]
    a[0] = a[0] @ col_perm_m
    c2 = hadamard_cube(a)
    c1_permed = c1[col_perm, :, :]
    assert np.allclose(c1_permed, c2)
    print("passed right permutation test: it permutes cube yz slices")

    a = a_orig.copy()
    c1 = hadamard_cube(a)
    row_perm = [1, 0, 3, 4, 2, 5]
    row_perm_m = np.eye(6)[row_perm, :]
    for i in range(3):
        a[i] = row_perm_m @ a[i]
    c2 = hadamard_cube(a)
    assert np.allclose(c1, c2)
    print("passed left (shared) permutation test: it does not change cube")

    # phase = np.exp(2j * np.pi * np.random.uniform() * np.ones(6))
    phase = np.exp(2j * np.pi * np.random.uniform(size=6))
    phase_m = np.diag(phase)

    a = a_orig.copy()
    c1 = hadamard_cube(a)
    for i in range(3):
        a[i] = phase_m @ a[i]
    verify_mub(a)
    c2 = hadamard_cube(a)
    assert np.allclose(c1, c2)
    print("passed left (shared) phase test: it does not change the cube")

    a = a_orig.copy()
    c1 = hadamard_cube(a)
    a[0] = a[0] @ phase_m
    c2 = hadamard_cube(a)
    assert np.allclose(c1, c2)
    print("passed right phase test: it does not change the cube")


# that's just a one-time validation of lemmata about the cube,
# need not run every time:
# test_cube_invariances(a) ; exit()




sx = c[0, :, :]
sy = c[:, 0, :]
sz = c[:, :, 0]
verify_hadamard(sx)
verify_hadamard(sy)
verify_hadamard(sz)

b1 = sy
b2 = sz

'''
for b in [sx, sy, sz]:
    print(angler(b))

f1 = get_canonizer(b1)
f2 = get_canonizer(b2)

assert f1 is not None
assert f2 is not None
print(f1)
print(f2)
'''

'''
prod = trans(b1, b2)
print(is_phase_equivalent(prod, sx))
'''

for oneidx, one in zip(["A", "B", "A^dag B"], [a[1], a[2], trans(a[1], a[2])]):
    for otheridx, other in zip(["sx", "sy", "sz"], [sx, sy, sz]):
        eq = equivalent_hadamard_brute_force(one, other, atol=1e-2)
        if eq is not None:
            print("equivalent:", oneidx, otheridx, eq)


exit()



slices = [c[0, :, :], c[:, 0, :], c[:, :, 0]]
for direction in range(3):
    for coord in range(6):
        # print("====")
        # print(angler(s))
        # print(visualize_clusters(s, group_conjugates=False))
        s = slic(c, direction, coord)
        result = find_blocks(s)
        if result is not None:
            print("F ", end='')
            continue
        result = find_blocks(s.T)
        if result is not None:
            print("FT ", end='')
            continue
        print("N ", end='')
        verify_hadamard(s)
        print()
        print(angler(s))
        exit()
    print()
exit()




slices = [c[0, :, :], c[:, 0, :], c[:, :, 0]]
for s in slices:
    # print("====")
    # print(angler(s))
    # print(visualize_clusters(s, group_conjugates=False))
    try:
        find_blocks(s)
        # print("^^^ Fourier")
        print("F ", end='')
        continue
    except:
        pass
    try:
        find_blocks(s.T)
        # print("^^^ Fourier transposed")
        print("FT ", end='')
        continue
    except:
        pass
    # print("^^^ not Fourier")
    print("N ", end='')
print()
exit()


visualize_3d(c) ; exit()

print(angler(c)) ; exit()

# visualize(c) ; exit()


print(visualize_clusters(c, group_conjugates=False)) ; exit()


indcs = np.arange(6, dtype=int)
coords0, coords1, coords2 = np.meshgrid(indcs, indcs, indcs)

plt.scatter(c.flatten().real, c.flatten().imag, s=(coords0.flatten() + 1) ** 2, c=coords1.flatten(), alpha=0.3)
plt.show()

exit()



prod = np.conjugate(a[0].T) @ a[1]
transfers = [a[0], a[1], prod, a[0].T, a[1].T, prod.T]
names = ["U1", "U2" "U3", "U1T", "U2T" "U3T"]
for name, transfer in zip(names, transfers):
    set_first, set_second = True, True

    d_left, p_left, x, y, p_right, d_right, dist = the_true_decomposition(a[i], set_first, set_second)
    d_l = phase_to_deg(np.diag(d_left))
    d_r = phase_to_deg(np.diag(d_right))
    p_l = np.argmax(p_left, axis=0) # permutation of rows, but not here.
    p_r = np.argmax(p_right, axis=0) # permutation of columns, but not here.
    print("filename", filename, "which", name, "D_l", d_l, "P_l", p_l, "x", phase_to_deg(x), "y", phase_to_deg(y), "P_r", p_r, "D_r", d_r, "distance", dist)
