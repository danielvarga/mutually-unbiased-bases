import numpy as np

from base import *

a = np.load("fortyeight_structured.npy")


# this was created from
np.random.seed(1)
def random_phases(shape):
    return np.exp(2 * np.pi * 1j * np.random.uniform(size=shape))
x, y = random_phases((2, ))
F = canonical_fourier(x, y)
print(x, y)



for i, b in enumerate(a):
    transfer = trans(F.T, b.T)
    transfer = np.conjugate(F) @ b.T
    print(i, find_partitions(transfer, atol=1e-4))
    print(i, find_partitions(transfer.T, atol=1e-4))
    print(i, get_canonizer(transfer))
