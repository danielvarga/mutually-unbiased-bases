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


'''
assert len(mubs.keys()) == 2
keys = sorted(mubs.keys())
factorized_mub_m = mubs[keys[0]]
factorized_mub_n = mubs[keys[1]]
print("qMUB M:", keys[0])
print("qMUB N:", keys[1])
assert len(factorized_mub_m) == len(factorized_mub_n)== 3
'''

assert len(mubs.keys()) == 1
factorized_mub = mubs[list(mubs.keys())[0]]


def remove_right_effects(factors):
    left_phases, left_pm, x, y, right_pm, right_phases = factors
    return left_phases, left_pm, x, y, np.eye(6), np.ones(6)


def remove_right_effects_from_mub(factorized_mub):
    factorized_mub2 = []
    for i in range(3):
        factorized_mub2.append(remove_right_effects(factorized_mub[i]))
    return factorized_mub2


def reorder_by_fourier_params(factorized_mub):
    def signature(x, y):
        # when this signature is in increasing order, we have (a, a), (a, 0), (-a, 0).
        return - np.angle(x) - np.angle(y)

        # when this signature is in increasing order, we have (a, 0), (-a, 0), (a, a).
        # that's the order of normalized/mub_120.npy
        return - np.angle(x) + 10 * np.angle(y)

    signatures = [signature(factors[2], factors[3]) for factors in factorized_mub]

    # signatures = [- np.angle(factors[2]) + 10 * np.angle(factors[3]) for factors in factorized_mub]


    perm = np.argsort(np.array(signatures))
    factorized_mub2 = [None, None, None]
    for to_pos, from_pos in enumerate(perm):
        factorized_mub2[to_pos] = factorized_mub[from_pos]

    # verify
    signatures2 = [signature(factors[2], factors[3]) for factors in factorized_mub2]
    perm = np.argsort(np.array(signatures2))
    assert perm.tolist() == list(range(3))

    return factorized_mub2


def switch_perm_phase(pm, phases):
    matrix = pm @ np.diag(phases)
    phases2 = np.sum(matrix, axis=1)
    matrix2 = np.diag(1 / phases2) @ matrix
    assert np.allclose(matrix, np.diag(phases2) @ matrix2)
    assert np.allclose(matrix2 * (matrix2 - 1), 0)
    pm2 = np.around(matrix2).real.astype(int)
    return phases2, pm2


def switch_phase_perm(phases, pm):
    phases2, pm2 = switch_perm_phase(pm.T, phases)
    return pm2.T, phases


def apply_left_phase(factorized_mub, left_phases_to_apply):
    factorized_mub2 = []
    for i in range(3):
        left_phases, left_pm, x, y, right_pm, right_phases = factorized_mub[i]
        left_phases_after = left_phases * left_phases_to_apply
        factorized_mub2.append((left_phases_after, left_pm, x, y, right_pm, right_phases))
    return factorized_mub2


def apply_left_permutation(factorized_mub, left_pm_to_apply):
    factorized_mub2 = []
    for i in range(3):
        left_phases, left_pm, x, y, right_pm, right_phases = factorized_mub[i]
        left_phases_after, pm2 = switch_perm_phase(left_pm_to_apply, left_phases)
        factorized_mub2.append((left_phases_after, pm2 @ left_pm, x, y, right_pm, right_phases))
    return factorized_mub2


def remove_first_left_effect(factorized_mub):
    factors_0 = factorized_mub[0]
    left_phases_0, left_pm_0 = factors_0[:2]
    factorized_mub = apply_left_phase(factorized_mub, 1 / left_phases_0)
    factorized_mub = apply_left_permutation(factorized_mub, left_pm_0.T)
    return factorized_mub


factorized_mub = remove_right_effects_from_mub(factorized_mub)
factorized_mub = reorder_by_fourier_params(factorized_mub)
factorized_mub = remove_first_left_effect(factorized_mub)


def dump_basis(factors):
    left_phases, left_pm, x, y, right_pm, right_phases = factors
    # left_pm, left_phases = switch_phase_perm(left_phases, left_pm)
    print(np.angle(left_phases) * 180 / np.pi)
    print(left_pm.astype(int))
    print(np.angle(x) * 180 / np.pi, np.angle(y) * 180 / np.pi)


def dump_mub(factorized_mub):
    for i in range(3):
        print("=====")
        print(i)
        dump_basis(factorized_mub[i])


def short_dump_mub(factorized_mub):
    # B_1 is F(a,0), we don't dump that.
    # we don't look into B_2 and B_3 permutations, not shown.
    np.set_printoptions(precision=12, suppress=True, linewidth=100000)
    def angler(x):
        return " ".join(map(str, (np.angle(x) * 180 / np.pi).tolist()))
    print(angler(factorized_mub[1][0]), angler(factorized_mub[2][0]))


short_dump_mub(factorized_mub)
exit()


A = 0.42593834
B = 0.35506058
C = 0.38501704
D = 0.40824829
numerical_magic_constants = [36 * c **2 for c in [A, B, C]]

def reconstruct_tripartition(b0, b1):
    magic_constants = [1, 2, 3]
    msquared = np.abs(np.conjugate(b0.T) @ b1) ** 2
    masks = [np.isclose(msquared, numerical_magic_constants[i]).astype(int) for i in range(3)]
    assert np.all(masks[0] | masks[1] | masks[2])
    magic_matrix = sum(magic_constants[i] * masks[i] for i in range(3))
    return magic_matrix, magic_constants

mub = []
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
            terms.append(closeness(np.abs(prod), target))
    return sum(terms)


for (i, j) in [(0, 1), (1, 2), (2, 0)]:
    prod = np.conjugate(mub[i].T) @ mub[j]
    print(f"| B_{i}^dagger B_{j} | / 6")
    print(np.abs(prod) / 6)


loss = loss_function(mub)
print("loss", loss)


# normalized/mub_120.npy fourier parameters:
# a, -b
# -a-b, -b
# a, a+b
# where a = 55.11476631781354, b = 0.007509388499017998
