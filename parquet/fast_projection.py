import numpy as np


n = 6
N = 4


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
    values = np.linalg.norm(allpairs.sum(axis=1), axis=1) ** 2
    assert values.shape == (num_total_gridpoints, batchsize)

    values /= n ** 3

    print("G min/avg/max", values.min(), values.mean(), values.max())

    output_shape = (num_unique_gridpoints, batchsize)
    sums = np.zeros(output_shape)
    counts = np.zeros(output_shape)
    for i in range(len(values)):
        which_in_bu = unique_inverse[i]
        sums[which_in_bu, :] += values[i, :]
        counts[which_in_bu, :] += 1

    averages = sums / counts
    assert averages.shape == output_shape
    print("Gbar min/avg/max", averages.min(), averages.mean(), averages.max())

    # -> averages[i] tells the Gbar value for gridpoint b_unique[i]
    return b_unique, averages


MUB = np.load("../data/triplet_mub_57638.npy")
H = MUB[0] * n ** 0.5
assert np.allclose(np.abs(H[0]), 1)

batchsize = 1000
Hs = np.array([H] * batchsize)

import time
start = time.perf_counter()
b_unique, averages = evaluate(Hs)
print(time.perf_counter() - start, f"seconds for {batchsize} {n}x{n} matrices, gridsize {2 * N - 1}^{n}, {len(b_unique)} unique gridpoints")
