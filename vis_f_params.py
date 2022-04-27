# cat data/classify.couts.all | python vis_f_params.py

import sys
import numpy as np

import matplotlib.pyplot as plt


# filename triplets/triplet_mub_00000.npy x1 41.92215214771241 y1 4.9202858965324445 x2 -161.94661689307344 y2 114.54937424571182 Dl2 [   0.        90.37614 -161.65514   98.47066 -148.03671  150.49671]

fs = []
for l in sys.stdin:
    a = l.replace("]", " ]").strip().split()
    if len(a) == 3:
        continue
    assert len(a) == 19, a
    assert (a[0], a[10]) == ("filename", "Dl2")
    f = list(map(float, a[3:11:2]))
    fs.append(f)

fs = np.array(fs)
print(fs.shape, fs.dtype)


plt.hexbin(fs[:, 0], fs[:, 1], gridsize=30)
plt.show()


plt.scatter(fs[:, 0], fs[:, 1], s=(fs[:, 2] + 180)/30, c=fs[:, 3])
plt.show()

low0, high0 = -64, -58
low1, high1 = -36, -29
fs_filtered = fs[(low0 < fs[:, 0]) & (fs[:, 0] < high0) & (low1 < fs[:, 1]) & (fs[:, 1] < high1)]

plt.scatter(fs_filtered[:, 2], fs_filtered[:, 3])
plt.show()
print(fs_filtered.shape)
print(fs_filtered[:, 0].max())
exit()


plt.scatter(fs[:, 0], fs[:, 1])
plt.show()


exit()


# from mpl_toolkits.mplot3d import axes3d
# fig = plt.figure()
# ax = fig.gca(projection='3d')



for f in fs[:1000]:
    x1, y1, x2, y2 = f
    # plt.plot([x1, x2], [y1, y2], [-180, 180])
    plt.plot([x1, x1], [y1, y1], [-180, 180])

plt.show()
