import sys
import numpy as np

np.set_printoptions(precision=5, suppress=True)

filename = sys.argv[1]

a = np.load(filename)

try:
    nm, n = a.shape
    m = nm // n
except:
    m, n, _ = a.shape

a = a.reshape((m, n, n))

assert n == 6 and m == 4

A = 0.42593834
B = 0.35506058
C = 0.38501704
D = 0.40824829 # 1 / 6 ^ 0.5
A_SYM, B_SYM, C_SYM, D_SYM = 2, 3, 4, 5
mapping = {0: 0, 1: 1, A: A_SYM, B: B_SYM, C: C_SYM, D: D_SYM}

def calculate_prods(a):
    prods = np.zeros((m, m, n, n), dtype=np.complex128)
    for i in range(m):
        for j in range(m):
            prods[i, j] = np.conjugate(a[i].T) @ a[j]

    for i in range(m):
        try:
            assert np.allclose(prods[i, i], np.eye(n), atol=1e-5)
        except:
            print("should be identity")
            print(prods[i, i])
            exit()

    prods_sym = np.zeros_like(prods, dtype=np.uint8)
    covered = np.zeros_like(prods, dtype=bool)

    for real, sym in mapping.items():
        covered_now = np.isclose(np.abs(prods), real, atol=1e-5)
        covered |= covered_now
        prods_sym += covered_now.astype(np.uint8) * sym
    assert np.all(covered), "you promised me that all magnitudes of product matrix elements are in a restricted set."

    return prods, prods_sym

prods, p = calculate_prods(a)

pivot = None
for i in range(m):
    allbuti = list(range(i)) + list(range(i+1, m))
    if np.all(p[i][allbuti] == D_SYM):
        pivot = i
        break

assert pivot is not None, "you promised me that one is MUB with all the others."

allbutpivot = list(range(pivot)) + list(range(pivot+1, m))

a = a[[pivot] + allbutpivot]

a0 = a[0].copy()
for i in range(m):
    a[i] = np.conjugate(a0.T) @ a[i]

assert np.allclose(a[0], np.eye(n))

for i in range(1, m):
    for j in range(n):
        a[i, :, j] /= a[i, 0, j] / np.abs(a[i, 0, j]) # rotating the columns by the negative phase of the first element

prods, p = calculate_prods(a)

if len(sys.argv) > 2:
    assert len(sys.argv) == 3, "at most two arguments are accepted"
    output_filename = sys.argv[2]
    np.save(output_filename, a)

print(p[1, 2])
print(p[2, 3])
print(p[3, 1])

exit()


import matplotlib.pyplot as plt


row_labels = (np.arange(n)[None, :, None] + np.zeros((m-1, n, n))).flatten()
col_labels = (np.arange(n)[None, None, :] + np.zeros((m-1, n, n))).flatten()

'''
plt.scatter(row_labels, (np.angle(a[1:]) / np.pi).flatten())
plt.show()
'''

plt.scatter(np.real(a[1:].flatten()), np.imag(a[1:].flatten()), c=row_labels, s=row_labels * 20)
b = a.copy()
b *= np.exp(1j*np.pi * 2 / 3) # 120 degrees phase rotation
plt.scatter(np.real(b[1:].flatten()), np.imag(b[1:].flatten()), c=row_labels, s=row_labels * 20, marker='x')

plt.show()

# np.save("optimum_in_gilyen_normal_form.npy", a)


for i in range(m):
    print(i, i)
    print(np.abs(np.conjugate(a[i].T) @ a[i]))

for i in range(m):
    for j in range(i + 1, m):
        print(i, j)
        print(np.abs(np.conjugate(a[i].T) @ a[j]))
