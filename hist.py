import sys
import matplotlib.pyplot as plt

a = list(map(float, sys.stdin.readlines()))
plt.hist(a, bins=360)
plt.show()
