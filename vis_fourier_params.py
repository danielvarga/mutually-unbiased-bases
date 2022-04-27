# cat data/classify.couts.all | python vis_fourier_params.py

import sys
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider


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


def nearby(x, y, delta=5):
    low0, high0 = x-delta, x+delta
    low1, high1 = y-delta, y+delta
    fs_filtered = fs[(low0 < fs[:, 0]) & (fs[:, 0] < high0) & (low1 < fs[:, 1]) & (fs[:, 1] < high1)]
    return fs_filtered


fig, ax = plt.subplots()

x = y = 0.0

dots = nearby(x, y)

l, = plt.plot(dots[:, 2], dots[:, 3], lw=0, marker='o')

ax = plt.axis([-180, 180, -180, 180])

x_ax = plt.axes([0.25, .03, 0.50, 0.02])
y_ax = plt.axes([0.25, .06, 0.50, 0.02])

# Slider
x_s = Slider(x_ax, 'x', -180, 180, valinit=0)
y_s = Slider(y_ax, 'y', -180, 180, valinit=0)

def update(val):
    x = x_s.val
    y = y_s.val
    # update curve
    dots = nearby(x, y)
    l.set_xdata(dots[:, 2])
    l.set_ydata(dots[:, 3])
    # redraw canvas while idle
    fig.canvas.draw_idle()

# call update function on slider value change
x_s.on_changed(update)
y_s.on_changed(update)

plt.show()


plt.hexbin(fs[:, 0], fs[:, 1], gridsize=30)
plt.show()


plt.scatter(fs[:, 0], fs[:, 1], s=(fs[:, 2] + 180)/30, c=fs[:, 3])
plt.show()
