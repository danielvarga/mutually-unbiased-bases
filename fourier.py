import sys
import itertools
import numpy as np

TIP = 2j * np.pi

# Fourier basis
def f(a, b):
    x = np.exp(TIP * a)
    y = np.exp(TIP * b)
    w = np.exp(TIP / 3)
    m = np.ones((6, 6), dtype=np.complex128)
    m[1::2, 1::2] *= -1
    m[1::3, 1::2] *= x
    m[2::3, 1::2] *= y
    mini = np.array([[w ** 2, w], [w, w ** 2]])
    # not matrix product
    m[1:3, 1:3] *= mini
    m[4:6, 1:3] *= mini
    m[1:3, 4:6] *= mini
    m[4:6, 4:6] *= mini
    m /= 6 ** 0.5
    return m


# generalized Fourier basis
def fg(a, b, do_dagger, perm1, phase1, perm2, phase2):
    m = f(a, b)
    if do_dagger:
        m = np.conjugate(m.T)
    return np.diag(phase1) @ perm1 @ m @ perm2 @ np.diag(phase2)


np.set_printoptions(precision=5, suppress=True)

'''
m1 = f(0.2, 0.33)
m2 = f(0.23, 0.3)
prod = np.conjugate(m1.T) @ m2
print(np.abs(prod))
'''

perm1 = np.eye(6)[np.random.permutation(6), :]
perm2 = np.eye(6)[np.random.permutation(6), :]
phase1 = np.exp(TIP * np.random.uniform(size=6))
phase2 = np.exp(TIP * np.random.uniform(size=6))

m = fg(0.22342134, 0.33234243, True, perm1, phase1, perm2, phase2)
# m = fg(0.22342134, 0.33234243, False, np.eye(6), np.ones(6), np.eye(6), np.ones(6)) # un-generalized, same as f(a, b)

for j in range(6):
    m[:, j] /= m[0, j] / np.abs(m[0, j]) # rotating the columns by the negative phase of the first element

print(np.angle(m) * 180 / np.pi)

for i1 in range(6):
    for i2 in range(6):
        print("=>", i1, i2)
        angles1 = np.angle(m[i1, :])
        angles2 = np.angle(m[i2, :])
        print((angles1[:, None] - angles2[None, :]) / np.pi * 180)


exit()

prod = np.conjugate(m.T) @ m
assert np.allclose(np.abs(prod - np.eye(6)), 0)


filename, = sys.argv[1:]

a = np.load(filename)

assert a.shape == (4, 6, 6)
assert np.all(a[0] == np.eye(6))

B_1 = a[1]

N = 100
unit_phase = np.ones(6)
for a in np.linspace(0, 1, N):
    print(a, file=sys.stderr)
    for b in np.linspace(0, 1, N):
        for perm1 in itertools.permutations(range(6)):
            perm1m = np.eye(6)[perm1, :]
            for perm2 in [list(range(6))]: # itertools.permutations(range(6)):
                perm2m = np.eye(6)[perm2, :]
                for do_dagger in [False, True]:
                    M = fg(a, b, do_dagger, perm1m, unit_phase, perm2m, unit_phase)
                    print(np.abs(B_1 - M).max(), a, b)
                    # print(a, b, perm1, perm2, do_dagger, np.linalg.norm(B_1 - M))
