import numpy as np
from itertools import permutations

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


TP = 2 * np.pi
TIP = 2j * np.pi
W = np.exp(TIP / 3)


def t(b1, b2):
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
            dist = np.linalg.norm(b1p - b2, ord=1)
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


x = np.exp(TIP * 0.57673)
y = np.exp(TIP * 0.23467)


def tests():
    b1 = canonical_fourier(x, y)
    b2 = canonical_fourier(y, x)
    b1[1:, 1:] = b1[5:0:-1, 5:0:-1]
    assert np.allclose(b1, b2)

    b1 = canonical_fourier(x, y)
    b2 = canonical_fourier(-x, y)
    b1[[1, 4], :] = b1[[4, 1], :]
    # aka b1 = b1[[0, 4, 2, 3, 1, 5], :]
    assert np.allclose(b1, b2)

    b1 = canonical_fourier(x, y)
    b2 = canonical_fourier(x, -y)
    b1[[2, 5], :] = b1[[5, 2], :]
    # aka b1 = b1[[0, 1, 5, 3, 4, 2], :]
    assert np.allclose(b1, b2)

    b1 = canonical_fourier(x, y)
    b2 = canonical_fourier(x * np.sqrt(W), y / np.sqrt(W))
    b1 = b1[[0, 4, 5, 3, 1, 2], :]
    b1 = b1[:, [0, 5, 2, 1, 4, 3]]
    assert np.allclose(b1, b2)

    # by definition it's the square of the previous, but let's see:
    # it only permutes the columns.
    b1 = canonical_fourier(x, y)
    b2 = canonical_fourier(x * W, y / W)
    b1 = b1[:, [0, 3, 2, 5, 4, 1]]
    assert np.allclose(b1, b2)
    print(t(b1, b2))

    # these can't be transformed
    b1 = canonical_fourier(x, y)
    b2 = canonical_fourier(x * W, y * W)
    result = t(b1, b2)
    assert result[-1] > 0.01


tests()

