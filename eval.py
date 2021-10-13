import sys
import numpy as np

filename, = sys.argv[1:]

a = np.load(filename)

try:
    nm, n = a.shape
    m = nm // n
except:
    m, n, _ = a.shape

a = a.reshape((m, n, n))
print(a.shape)


for i in range(1, m):
    print(i, "=>")
    print(np.angle(a[i] * 6 ** 0.5))
exit()

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

