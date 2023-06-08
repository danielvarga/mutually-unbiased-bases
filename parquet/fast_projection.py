import numpy as np


n = 6
N = 4



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


x = np.arange(-N+1, N, dtype=int)
xs = [x] * n
# nth direct power of [-N, N] interval:
a = np.meshgrid(*xs)
# array of all possible [-N, N] intervals:
b = np.stack(a, axis=-1).reshape((-1, n))
# ...that sum to 0:
b = b[b.sum(axis=-1) == 0]

print(b.shape)

# each re-ordered descending:
b_ordered = - np.sort(- b, axis=-1)
# keeping only the distinct ones:
b_unique, unique_inverse = np.unique(b_ordered, axis=0, return_inverse=True)
# -> b_unique[unique_inverse] == b_ordered

print(b_unique.shape)


# assumes global variables b, b_unique, unique_inverse
# b is (num_total_gridpoints, n), Hs is (batchsize, n, n)
def evaluate(Hs):
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
    # print(list(zip(values, b))[:10])

    '''
    for value, bb in zip(values, b):
        if np.isclose(value, 0):
            print(value, bb)
    '''

    # print("G min/avg/max", values.min(), values.mean(), values.max())

    output_shape = (num_unique_gridpoints, batchsize)
    sums = np.zeros(output_shape)
    counts = np.zeros(output_shape)
    for i in range(len(values)):
        which_in_bu = unique_inverse[i]
        sums[which_in_bu, :] += values[i, :]
        counts[which_in_bu, :] += 1

    averages = sums / counts
    assert averages.shape == output_shape
    # print("Gbar min/avg/max", averages.min(), averages.mean(), averages.max())

    # -> averages[i] tells the Gbar value for gridpoint b_unique[i]
    return b_unique, averages


'''
MUB = np.load("../data/triplet_mub_57638.npy")
H = MUB[0] * n ** 0.5
assert np.allclose(np.abs(H[0]), 1)
'''

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


cubes = np.load("all_straight_cubes.npy")
cube = cubes[0]
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
    b_unique, averages = evaluate(H[None, :, :])
    assert np.allclose(b_unique[-1], [3, 3, 3, -3, -3, -3]), b_unique[-1]
    print(averages[-1], mini33(H[None, :, :], 3))
exit()



'''
print(H)
cols = H.sum(axis=0)
print(cols)
print(np.linalg.norm(cols) ** 2 / 36)
H[[3, 4, 5], :] = 1 / H[[3, 4, 5], :]
print("====")
print(H)
cols = H.sum(axis=0)
print(cols)
print(np.linalg.norm(cols) ** 2 / 36)
exit()
'''

batchsize = 1
Hs = np.array([H] * batchsize)

import time
start = time.perf_counter()
b_unique, averages = evaluate(Hs)
print(time.perf_counter() - start, f"seconds for {batchsize} {n}x{n} matrices, gridsize {2 * N - 1}^{n}, {len(b_unique)} unique gridpoints")
