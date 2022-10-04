import numpy as np


n = 6
N = 8

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
variables = set(tuple(var) for var in b_unique)

print(b_unique.shape)

MUB = np.load("../data/triplet_mub_57638.npy")
H = MUB[0] * n ** 0.5
assert np.allclose(np.abs(H), 1)

# gridpoint x row x column
allpairs = H[None, :, :, ] ** b[:, :, None]
values = np.linalg.norm(allpairs.sum(axis=1), axis=1) ** 2
# -> indexed with gridpoints
print(values.shape)

values /= n ** 3

print("G min/avg/max", values.min(), values.mean(), values.max())

sums = np.zeros(len(b_unique))
counts = np.zeros(len(b_unique))
for i in range(len(values)):
    which_in_bu = unique_inverse[i]
    sums[which_in_bu] += values[i]
    counts[which_in_bu] += 1

averages = sums / counts
print("Gbar min/avg/max", averages.min(), averages.mean(), averages.max())

# -> averages[i] tells the Gbar value for gridpoint b_unique[i]
