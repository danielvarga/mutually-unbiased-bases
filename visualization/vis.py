import sys
import itertools
from base import *

np.set_printoptions(precision=3, suppress=True, linewidth=100000)


def interlace_cube(c):
    c = c[:, [0, 3, 1, 4, 2, 5], :]
    c = c[:, :, [0, 3, 1, 4, 2, 5]]
    return c


# this only works for the Szollosi aka 0 axis, and that's good for now
def permute_cube(c, p1, p2):
    c = c[:, p1, :]
    c = c[:, :, p2]
    return c


cube_file, = sys.argv[1:]

c = np.load(cube_file)

verify_cube_properties(c)


def is_hadamard(mini):
    assert mini.shape == (2, 2)
    d = mini[0, 0] * mini[1, 1] + mini[0, 1] * mini[1, 0]
    return np.isclose(d, 0)


def normalize_szollosi(m_orig):
    perms = list(itertools.permutations(range(6)))
    for p1 in perms:
        for p2 in perms:
            m = m_orig[list(p1), :]
            m = m[:, list(p2)]
            m1 = m[:2, :2] ; m2 = m[2:4, 2:4] ; m3 = m[4:, 4:]
            if is_hadamard(m1) and np.allclose(m1, m2) and np.allclose(m1, m3):
                print("!!!\n", angler(m))
                return list(p1), list(p2)
    raise Exception("permutation not found")




def find_minicubes(c):
    pairs = list(itertools.combinations(range(6), 2))
    for xp in pairs:
        for yp in pairs:
            for zp in pairs:
                m = c.copy()
                m = m[xp, :, :]
                m = m[:, yp, :]
                m = m[:, :, zp]
                if is_hadamard(m[0, :, :]):
                    correct = True
                    for axis in (0, 1, 2):
                        for coord in (0, 1):
                            slc = slic(m, axis, coord)
                            if not is_hadamard(slc):
                                correct = False
                    if correct:
                        print(xp, yp, zp)
                        # if xp != (0,3) or yp != (0,3): continue
                        '''
                        for x in xp:
                            for y in yp:
                                for z in zp:
                                    print(x, y, z)
                        '''


c = c[:, [0,3,1,4,2,5], :]

find_minicubes(c) ; exit()


def build_pair(m):
    m = m[[1,0,3,2,5,4], :]
    m = m[:, [1,0,3,2,5,4]]
    m = m.T
    m = np.conjugate(m)
    return m


vis = visualize_clusters(c)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
for i in range(6):
    for j in range(6):
        for k in range(6):
            if i > 0 and j > 0 and k > 0:
                continue
            ax.scatter(i, j, k, color='b', s=1)
            ax.text(i, j, k, str(vis[i, j, k]), size=10, zorder=1, color='k') 

plt.show()
exit()







p1, p2 = normalize_szollosi(c[0, :, :])
c = permute_cube(c, p1, p2)
verify_cube_properties(c)

m = c[0, :, :]
print(angler(m))
m1 = m[:2, :2] ; m2 = m[2:4, 2:4] ; m3 = m[4:, 4:]
if is_hadamard(m1) and np.allclose(m1, m2) and np.allclose(m1, m3):
    print("successfully normalized")
else:
    assert False



c = c[:, :, [0, 1, 2, 5, 4, 3]]

m = c[0, :, :]
m_prime = build_pair(m)

print("sz0", angler(m))
# print("sz0d", angler(m_prime))
print("sz1", angler(c[1, :, :]))
exit()


print("====")
print(c.shape)
print(angler(c))



for i in range(6):
    print(np.allclose(m_prime, c[i, :, :]))

exit()


axis = 0
coord = 0

a = np.zeros((6, 6), dtype=int)
found = 0
for i in range(6):
    for j in range(6):
        for k in range(6):
            for l in range(6):
                if i >= j or k >= l:
                    continue

                m = slic(c, axis, coord)
                m = m[[i, j]]
                m = m[:, [k, l]]
                if is_hadamard(m):
                    for x in (i, j):
                        for y in (k, l):
                            a[x, y] += 1
                    found += 1
                    '''
                    print(found)
                    print(a)
                    print()'''
                    print(i, j, k, l)
                    print(angler(m))
                    print()
                # print(i, j, k, l, is_hadamard(m))


exit()






k = 0
for i in range(3):
    for j in range(3):
        mini = c[k, 2*i:2*i+2, 2*j:2*j+2]
        print(mini[0, 0] * mini[1, 1] - mini[0, 1] * mini[1, 0])
        # print(angler(mini))

print(angler(c[0, :, :]))
