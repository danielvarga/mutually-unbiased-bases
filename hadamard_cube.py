import sys
import numpy as np
from itertools import permutations
import matplotlib.pyplot as plt


TP = 2 * np.pi
TIP = 2j * np.pi
W = np.exp(TIP / 3)



# x and y are phases. this is exactly formula (3) in https://arxiv.org/pdf/0902.0882.pdf
# not sure about numerical precision when doing (sixth ** 5) instead of -W.
def canonical_fourier(x, y):
    ws = np.ones((6, 6), dtype=np.complex128)
    ws[1:6:3, 1:6:2] *= x
    ws[2:6:3, 1:6:2] *= y
    sixth = - np.conjugate(W)
    for i in range(1, 6):
        for j in range(1, 6):
            ws[i, j] *= sixth ** ((i * j) % 6)
    return ws


def transform(b1, b2):
    ps = list(permutations(range(6)))
    bestdist = 1e9
    bestp1 = None
    bestp2 = None
    for p1 in ps:
        for p2 in ps:
            b1p = b1[p1, :]
            b1p = b1p[:, p2]
            b1p /= b1p[0:1, :]
            b1p /= b1p[:, 0:1]
            dist = np.abs(b1p - b2).sum()
            if dist < bestdist:
                bestdist = dist
                bestp1 = p1
                bestp2 = p2
    # phases are not provided, because i've only encountered
    # F(x,y) -> F(x', y') transformations where both the left and the
    # right diagonals were the identity matrix.
    # but they are easy to get from the returned data.
    # never tested, tbh:
    b1p = b1[bestp1, :]
    b1p = b1p[:, bestp2]
    colphases = b1p[0, :]
    b1p /= b1p[0:1, :]
    rowphases = b1p[:, 0]
    return bestp1, bestp2, bestdist


def the_true_decomposition(b_orig, set_first, set_second):
    atol = 1e-4
    b = b_orig.copy()
    assert np.allclose(np.abs(b), np.ones_like(b), atol=atol), "b should be sqrt(6) times a Hadamard basis."
    d_left1 = b[:, 0].copy()
    b /= b[:, 0:1]
    d_right1 = b[0, :].copy()
    b /= b[0:1, :]
    b_reconstruct1 = np.diag(d_left1) @ b @ np.diag(d_right1)
    assert np.allclose(b_reconstruct1, b_orig, atol=atol)

    # we are looking for elements that are far from ALL of 1, -1, W, W^2.
    mask = np.abs(b - 1) * np.abs(b - W) * np.abs(b - W ** 2) * np.abs(b + 1)
    # our goal is two find two such elements so that they are not the negative of each other.
    # that has many solutions, but any will suffice as (x, y).
    # this can probably fail when x and y are 6th roots of unity, or very close, but what the heck.
    ind = np.unravel_index(np.argsort(-mask, axis=None), mask.shape)
    # for any i, mask[ind[0][i], ind[1][j]] is the element that's the i-th largest in the whole 2d array.
    # https://stackoverflow.com/a/64338853/383313
    x = b[ind[0][0], ind[1][0]]
    col_x = ind[1][0]
    # let's take another element from x's column that is far from -x.
    # namely, the next possible one in this ordering:
    for i in range(1, len(ind[0])):
        col_y = ind[1][i]
        if col_y == col_x:
            y = b[ind[0][i], ind[1][i]]
            if not np.isclose(-y, x):
                break

    if x.imag < 0:
        x = np.conjugate(x)
    assert set_first or set_second, "set at least one of them"
    first = x if set_first else 1
    second = x if set_second else 1

    candidate_basis = canonical_fourier(first, second)
    p1, p2, dist = transform(candidate_basis, b)
    b_reconstruct2 = candidate_basis[p1, :]
    b_reconstruct2 = b_reconstruct2[:, p2]
    d_left2 = np.conjugate(b_reconstruct2[:, 0])
    b_reconstruct2 /= b_reconstruct2[:, 0:1]
    d_right2 = np.conjugate(b_reconstruct2[0, :])
    b_reconstruct2 /= b_reconstruct2[0:1, :]
    assert np.allclose(b_reconstruct2, b, atol=atol), f"{b_reconstruct2}\n{b}"

    p_left = np.eye(6)[p1, :]
    p_right = np.eye(6)[:, p2]
    b_reconstruct3 = np.diag(d_left2) @ p_left @ candidate_basis @ p_right @ np.diag(d_right2)
    assert np.allclose(b_reconstruct3, b, atol=atol)

    # b is not to be confused with b_orig. now we have to compose everything to get b_orig:
    d_left = np.diag(d_left1) @ np.diag(d_left2)
    d_right = np.diag(d_right2) @ np.diag(d_right1)
    b_reconstruct4 = d_left @ p_left @ candidate_basis @ p_right @ d_right
    assert np.allclose(b_orig, b_reconstruct4, atol=atol) # b_orig now!
    return d_left, p_left, first, second, p_right, d_right, dist


def phase_to_deg(x):
    return np.angle(x) / np.pi * 180


def verify_hadamard(b):
    n = len(b)
    assert np.allclose(np.abs(b) ** 2, 1 / n, atol=1e-4)
    prod = np.conjugate(b.T) @ b
    assert np.allclose(prod, np.eye(n), atol=1e-4)


def gently_verify_hadamard(b):
    n = len(b)
    print(np.abs(b) ** 2 * n - 1)
    prod = np.conjugate(b.T) @ b
    print(prod - np.eye(n))


def verify_sum(v):
    assert np.isclose(v.sum(), 1, atol=2e-4)


def angler(x):
    return np.angle(x) * 180 / np.pi


def hadamard_cube(a, pad_with_id=True):
    n = 6
    if pad_with_id:
        a = np.stack([np.eye(n, dtype=np.complex128), a[0], a[1]])
    assert a.shape == (3, n, n)
    # vectorize maybe? who cares?
    c = np.zeros((n, n, n), dtype=np.complex128)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                c[i, j, k] = \
                    (np.conjugate(a[0, :, i]) @ a[1, :, j]) * \
                    (np.conjugate(a[1, :, j]) @ a[2, :, k]) * \
                    (np.conjugate(a[2, :, k]) @ a[0, :, i])
    return 6 * c


# np.set_printoptions(precision=12, suppress=True, linewidth=100000)
np.set_printoptions(precision=5, suppress=True)

# triplet_mub_015.npy
filename = sys.argv[1]
a = np.load(filename)

for i in range(len(a)):
    verify_hadamard(a[i])

'''
for i in range(len(a)):
    print("unitary", i)
    print(np.conjugate(a[i].T) @ a[i])

for i in range(len(a)):
    print("unbiased to Id", i)
    print(np.abs(a[i]))

print("unbiased to each other")
print(np.abs(np.conjugate(a[0].T) @ a[1]))
'''

def verify_cube_properties(c):
    for i in range(6):
        # print("verifying 2D slices", i)
        verify_hadamard(c[i, :, :])
        verify_hadamard(c[:, i, :])
        verify_hadamard(c[:, :, i])

    for i in range(6):
        for j in range(6):
            # print("verifying 1D slices", i, j)
            verify_sum(c[i, j, :])
            verify_sum(c[:, i, j])
            verify_sum(c[j, :, i])


def visualize(c):
    from matplotlib.patches import Circle
    from matplotlib.collections import PatchCollection
    from matplotlib.lines import Line2D
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


c = hadamard_cube(a)
verify_cube_properties(c)


print(angler(a))
exit()


# dead code, because there are NO nice angles.
def nice_angles(c):
    div = c[:, None] / c[None, :]
    n = 1
    dn = div ** n
    # these are all simply repeat elements, first roots of unity:
    print("nice ratios between Hadamard cube elements", np.isclose(dn, 1, atol=1e-4).astype(int).sum())


# nice_angles(c) ; exit()

print(angler(c))

visualize(c) ; exit()


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
