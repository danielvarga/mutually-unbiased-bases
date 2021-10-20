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

'''
np.set_printoptions(threshold=sys.maxsize)
aa = np.angle(a[1:, 1:, :].flatten())
print((aa[:, None] - aa[None, :]) / np.pi * 180)
exit()
'''

for i in range(1, m):
    print(i, "=>")
    print(np.angle(a[i] * 6 ** 0.5) / np.pi * 180)
    angles = np.angle(a[i, 4, :])
    print(angles.shape)
    # print(np.pi / (angles[:, None] - angles[None, :]))
    print((angles[:, None] - angles[None, :]) / np.pi * 180)

exit()

import matplotlib.pyplot as plt

'''
for i in range(1, m):
    xs = []
    ys = []
    for j in range(n):
        for k in range(n):
            plt.arrow(2 * j, 2 * k, np.real(a[i, j, k]), np.imag(a[i, j, k]), head_width=0.1)
            # xs.append(2*j + np.real(a[i, j, k]))
            # ys.append(2*k + np.imag(a[i, j, k]))
    # plt.scatter(xs, ys)
    plt.title(f"B_{i}")
    plt.show()
'''

'''
print(np.abs(np.conjugate(a[1, :, :]).T.dot(a[2, :, :])))
print("---")
print(np.abs(np.conjugate(a[1, :, 2]).dot(a[2, :, 3])))
exit()
'''

from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D


A = 0.42593834
B = 0.35506058
C = 0.38501704
D = 0.40824829


cmap = plt.cm.viridis


for i in range(1, m):
    for j in range(i, m):
        print(f"B{i}† B{j}")
        fig, ax = plt.subplots()
        patches = []
        colors = []
        for row in range(n):
            for col in range(n):
                centre = 1 * row + 1j * col
                z = centre
                scalprod = 0
                for k in range(n):
                    use_segments = False
                    if not use_segments:
                        z = centre
                    delta = np.conjugate(a[i, k, row]) * a[j, k, col]
                    ax.arrow(np.real(z), np.imag(z), np.real(delta), np.imag(delta), head_width=0.02, length_includes_head=True)
                    z += delta
                    scalprod += delta
                z -= centre
                radius = np.abs(scalprod)
                circle = Circle((centre.real, centre.imag), radius)
                color = -1
                if np.isclose(radius, A):
                    color = A
                elif np.isclose(radius, B):
                    color = B
                elif np.isclose(radius, C):
                    color = C
                else:
                    pass
                    # assert False, "you promised me a product with element magnitudes in {a, b, c}"
                patches.append(circle)
                colors.append(color)
                print("%.5f" % np.abs(scalprod), end="\t")
            print()
        p = PatchCollection(patches, alpha=0.4, cmap=cmap)
        p.set_array(np.array(colors))
        ax.add_collection(p)

        custom_lines = [Line2D([0], [0], color=cmap(A), lw=4),
                Line2D([0], [0], color=cmap(B), lw=4),
                Line2D([0], [0], color=cmap(C), lw=4)]
        ax.legend(custom_lines, [f"a = {A}", f"b = {B}", f"c = {C}"])

        fig.colorbar(p)

        plt.title(r"$B_%d^\dag B_%d$" % (i, j))
        plt.show()

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

