# input: three lines of canonized_mubs, like
# cat canonized_mubs | grep mub_100.npy | python analyze_canonized_mubs.py

import sys
import numpy as np
import matplotlib.pyplot as plt

# filename normalized/mub_100.npy i 1 D_l [  -0.      -26.7955 -173.147   -38.4863 -168.1707 -150.0384] P_l [0 5 2 3 1 4] x 55.11879687165239 y 0.0005514574849221608 P_r [4 1 2 3 0 5] D_r [0. 0. 0. 0. 0. 0.] distance 4.003481295679986e-06


TP = 2 * np.pi
TIP = 2j * np.pi
W = np.exp(TIP / 3)


def degrees_to_phases(d):
    return np.exp(1j * np.pi / 180 * d)


def perm_to_pm(p):
    return np.eye(6)[p, :]


# TODO build a lib, do not copypaste this function everywhere.
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


def factors_to_basis(factors):
    left_phases, left_pm, x, y, right_pm, right_phases = factors
    return np.diag(left_phases) @ left_pm @ canonical_fourier(x, y) @ right_pm @ np.diag(right_phases)


def serialized_to_factors(serialized_basis):
    left_degrees, left_permutation, x_degrees, y_degrees, right_permutation, right_degrees = serialized_basis
    x = degrees_to_phases(x_degrees)
    y = degrees_to_phases(y_degrees)
    left_phases = degrees_to_phases(left_degrees)
    right_phases = degrees_to_phases(right_degrees)

    # beware of the T here!!! left and right permutation matrices
    # are represented differently as permutations.
    left_pm = perm_to_pm(left_permutation).T
    right_pm = perm_to_pm(right_permutation)

    factors = (left_phases, left_pm, x, y, right_pm, right_phases)
    basis = factors_to_basis(factors)
    assert np.allclose(np.conjugate(basis.T) @ basis, 6 * np.eye(6)), "does not seem to be a basis: " + str(np.conjugate(basis.T) @ basis)
    assert np.allclose(np.abs(basis), 1), "does not seem to be Hadamard: " + str(np.abs(basis))
    return factors


mubs = {}
for l in sys.stdin:
    l = l.replace("[", "").replace("]", "")
    a = l.strip().split()
    assert len(a) == 38
    filename = a[1]
    basis = int(a[3])
    left_degrees = np.array(list(map(float, a[5: 5 + 6])))
    right_degrees = np.array(list(map(float, a[30: 30 + 6])))
    left_permutation = np.array(list(map(int, a[12: 12 + 6])))
    right_permutation = np.array(list(map(int, a[23: 23 + 6])))
    assert a[18] == "x"
    x_degrees = float(a[19])
    y_degrees = float(a[21])
    if filename not in mubs:
        mubs[filename] = []
    factors = serialized_to_factors((left_degrees, left_permutation, x_degrees, y_degrees, right_permutation, right_degrees))
    mubs[filename].append(factors)


# later we will probably generalize
assert len(mubs.keys()) == 1
factorized_mub = mubs[filename]
assert len(factorized_mub) == 3


mub = []
matrices = [] # rows are tuples of 
for i in range(3):
    basis = factors_to_basis(factorized_mub[i])
    mub.append(basis)
mub = np.array(mub)


def closeness(a, b):
    return np.sum(np.abs(a - b) ** 2)


# numpy code reimplementing the tf code search.py:loss_fn()
def loss_function(mub):
    terms = []
    for u in mub:
        prod = np.conjugate(u.T) @ u / 6
        terms.append(closeness(prod, np.eye(6)))

    target = 1 / 6 ** 0.5
    for i in range(3):
        for j in range(i + 1, 3):
            # / 6 because what we call basis is 6**2 times what search.py calls basis.
            prod = np.conjugate(mub[i].T) @ mub[j] / 6
            print(">", np.abs(prod))
            terms.append(closeness(np.abs(prod), target))
    return sum(terms)


for (i, j) in [(0, 1), (1, 2), (2, 0)]:
    prod = np.conjugate(mub[i].T) @ mub[j]
    print(f"| B_{i}^dagger B_{j} | / 6")
    print(np.abs(prod) / 6)


loss = loss_function(mub)
print("loss", loss)
