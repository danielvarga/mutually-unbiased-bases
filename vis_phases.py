# input: canonized_mubs

import sys
import numpy as np
import matplotlib.pyplot as plt

# filename normalized/mub_100.npy i 1 D_l [  -0.      -26.7955 -173.147   -38.4863 -168.1707 -150.0384] P_l [0 5 2 3 1 4] x 55.11879687165239 y 0.0005514574849221608 P_r [4 1 2 3 0 5] D_r [0. 0. 0. 0. 0. 0.] distance 4.003481295679986e-06

mubs = {}
for l in sys.stdin:
    l = l.replace("[", "").replace("]", "")
    a = l.strip().split()
    assert len(a) == 38
    filename = a[1]
    basis = int(a[3])
    degrees = np.array(list(map(float, a[5:11])))
    assert a[18] == "x"
    x = float(a[19])
    y = float(a[21])
    if filename not in mubs:
        mubs[filename] = []
    mubs[filename].append((basis, degrees, x, y))

cool = []

for filename, mub in mubs.items():
    if len(mub) != 3:
        print(filename, "partially reconstructed, dropped")
        continue
    cool_index = None
    boring_count = 0
    permutation = None
    for i in range(3):
        basis, degrees, x, y = mub[i]
        if np.abs(degrees).sum() < 1e-3:
            # print("that's the weird one with zero phases, dropped for now")
            cool_index = None
            break
        if np.allclose(np.sort(degrees), np.array([-120, -120, 0, 0, 120, 120])):
            boring_count += 1
            permutation = np.argsort(degrees)
        else:
            cool_index = i
    if cool_index is None:
        continue
    assert boring_count == 2, mub
    cool_degrees = mub[cool_index][1]
    cool_degrees = cool_degrees[permutation]
    cool.append(cool_degrees)
    print(cool_degrees[:, None] - cool_degrees[None, :])

while len(cool) < 49:
    cool.append(np.zeros_like(cool_degrees))
    print("adding dummy data")

cool = np.array(cool).reshape((7, 7, 6))


cool_phases = np.exp(1j * np.pi / 180 * cool)

x = np.linspace(0, 30, 7)
y = np.linspace(0, 30, 7)
xx, yy = np.meshgrid(x, y)
xx = xx[:, :, None]
yy = yy[:, :, None]

plt.scatter(xx + np.real(cool_phases), yy + np.imag(cool_phases), s=4)
plt.show()
