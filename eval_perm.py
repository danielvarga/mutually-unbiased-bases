import sys
import numpy as np

np.set_printoptions(precision=5, suppress=True)

filename, = sys.argv[1:]

a = np.load(filename)

try:
    nm, n = a.shape
    m = nm // n
except:
    m, n, _ = a.shape

a = a.reshape((m, n, n))

perm = np.eye(n) # [[1, 2, 0, 3, 4, 5], :]

for i in range(m):
    # that does not even change |X_i^\dag X_j|. but you have to apply it to all m matrices.
    a[i] = perm @ a[i]

    # that applies the permutation to both the rows and the columns of |X_i^\dag X_j|.
    # you can use a different perm for each a[i].
    # a[i] = a[i] @ perm



a0 = a[3].copy()
for i in range(m):
    a[i] = np.conjugate(a0.T) @ a[i]

for i in range(1, m):
    for j in range(n):
        a[i, :, j] /= a[i, 0, j] / np.abs(a[i, 0, j]) # rotating the columns by the negative phase of the first element

a[0] = np.eye(6)

for i in range(m):
    print(f"==== {i}")
    print(np.angle(a[i]) / np.pi)

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

exit()

x = a[0]

# residual, see https://people.eecs.berkeley.edu/~wkahan/Math128/NearestQ.pdf
y = np.conjugate(x.T) @ x - np.eye(n)
# awesome approximation of unitary matrix closest to x wrt Frobenius norm
xhat = x - x @ y @ ( 1/2 * np.eye(n) - 3/8 * y + 5/16 * y @ y - 35/128 * y @ y @ y @ y)
# simpler, still pretty good approximation of the same:
xhat = x - 1/2 * x @ y

print(np.abs(np.conjugate(xhat.T) @ xhat))

