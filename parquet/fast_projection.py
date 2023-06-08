import numpy as np


TP = 2 * np.pi
TIP = 2j * np.pi
W = np.exp(TIP / 3)


def angler(M):
    return np.around(np.angle(M) / 2 / np.pi * 360, 1)


# Returns canonical fourier matrix as a complex Hadamard matrix.
def canonical_fourier(x, y):
    ws = np.ones((6, 6), dtype=np.complex128)
    ws[1:6:3, 1:6:2] *= x
    ws[2:6:3, 1:6:2] *= y
    sixth = - np.conjugate(W)
    for i in range(1, 6):
        for j in range(1, 6):
            ws[i, j] *= sixth ** ((i * j) % 6)
    return ws


# a,b degrees, returns complex Fourier matrix
def deg_fourier(a, b):
    x = np.exp(2 * np.pi * 1j * a / 360)
    y = np.exp(2 * np.pi * 1j * b / 360)
    return canonical_fourier(x, y)


def random_fouriers(n):
    fouriers = []
    for i in range(n):
        x, y = np.random.uniform(0, 180), np.random.uniform(0, 180)
        fouriers.append(deg_fourier(x, y))
    return np.array(fouriers)


def get_szollosis(n):
    import os
    directory = '../szollosis.all/'
    szollosi_collection = [np.load(directory + filename) for filename in os.listdir(directory)[:n]]
    return np.array(szollosi_collection)



def build_grid_object(n, N, slice_sum):
    x = np.arange(-N+1, N, dtype=int)
    xs = [x] * n
    # nth direct power of [-N, N] interval:
    a = np.meshgrid(*xs)
    # array of all possible [-N, N] intervals:
    b = np.stack(a, axis=-1).reshape((-1, n))
    # ...that sum to 0 or 1:
    b = b[b.sum(axis=-1) == slice_sum]

    print(b.shape)

    # each re-ordered descending:
    b_ordered = - np.sort(- b, axis=-1)
    # keeping only the distinct ones:
    b_unique, unique_inverse = np.unique(b_ordered, axis=0, return_inverse=True)
    # -> b_unique[unique_inverse] == b_ordered

    print(b_unique.shape)
    return (b, b_unique, unique_inverse)


# b is (num_total_gridpoints, n), Hs is (batchsize, n, n)
def evaluate(Hs, grid_object, do_averaging):
    (b, b_unique, unique_inverse) = grid_object

    num_total_gridpoints = len(b)
    num_unique_gridpoints = len(b_unique)
    assert b.shape == (num_total_gridpoints, n)
    batchsize = len(Hs)
    Hs = np.transpose(Hs, (1, 2, 0))
    assert Hs.shape == (n, n, batchsize)

    allpairs = Hs[None, :, :, :] ** b[:, :, None, None]
    assert allpairs.shape == (num_total_gridpoints, n, n, batchsize) # there's our memory bottleneck
    vs = allpairs.prod(axis=1).sum(axis=1)
    values = np.abs(vs) ** 2
    assert values.shape == (num_total_gridpoints, batchsize)

    values /= n ** 2
    if not do_averaging:
        return b, values

    output_shape = (num_unique_gridpoints, batchsize)
    sums = np.zeros(output_shape)
    counts = np.zeros(output_shape)
    for i in range(len(values)):
        which_in_bu = unique_inverse[i]
        sums[which_in_bu, :] += values[i, :]
        counts[which_in_bu, :] += 1

    averages = sums / counts
    assert averages.shape == output_shape
    # -> averages[i] tells the Gbar value for gridpoint b_unique[i]
    return b_unique, averages


def mini33(H_orig, k):
    H = H_orig.copy()
    assert H.shape == (1, 6, 6)
    assert np.allclose(np.abs(H), 1)
    H[:, :3, :] **= k
    H[:, 3:, :] **= -k
    prods = np.prod(H, axis=1)
    sums = np.sum(prods, axis=1)
    h = np.abs(sums) ** 2 / 6 / 6
    return h



n = 6
N = 4
slice_sum = 1
do_averaging = True
target_value = 1/6


grid_object = build_grid_object(n=n, N=N, slice_sum=slice_sum)
(b, b_unique, unique_inverse) = grid_object # but we don't use them below, only in evaluate().

cubes = np.load("all_straight_cubes.npy")
cube = cubes[2]
cube *= 6 ** 0.5

'''
Hs = get_szollosis(100)
Hs = random_fouriers(100) / 6 ** 0.5
H = Hs[0]
H = H[[3, 2, 5, 1, 0, 4], :]
H = H[:, [3, 1, 4, 2, 0, 5]]
H = H * 6 ** 0.5
'''

# for H in [H]:
for H in (cube[0, :, :], cube[0, :, :].T, cube[:, 0, :], cube[:, 0, :].T, cube[:, :, 0], cube[:, :, 0].T):
    gridpoints, values = evaluate(H[None, :, :], grid_object, do_averaging=do_averaging)

    print(f"{target_value} ratio", (np.isclose(values, target_value)).astype(int).sum(), "out of", len(values))
    for gridpoint, value in zip(gridpoints, values):
        if np.isclose(value, target_value, atol=1e-6): print(gridpoint, value)
    print("=====")
