import sys
import itertools
import numpy as np
import matplotlib.pyplot as plt


def angler(x):
    return np.angle(x) * 180 / np.pi


def collect_cube_npys():
    cs = []
    for l in sys.stdin:
        c = np.load(l.strip())
        cs.append(c)

    cs = np.array(cs)
    print(cs.shape)
    assert cs.shape[1:] == (6, 6, 6)

    np.save("all_straight_cubes.npy", cs)


def collect_perms(gamma):
    perms = []
    for gamma_permuted in itertools.permutations(gamma):
        gamma_permuted = np.array(gamma_permuted)
        perms.append(gamma_permuted)
    perms = np.array(perms, dtype=int)
    perms, cnts = np.unique(perms, axis=0, return_counts=True)
    assert cnts.max() == cnts.min()
    # -> is constant so we can remove duplicates
    return perms, cnts.min()


def visualize_torus(cs):
    selection = cs[:, :, :, :] # maybe not the whole, we could limit it here
    selection = selection.reshape((-1, 6))
    angles = angler(selection)

    # plt.scatter(angles[:, 0], angles[:, 1], alpha=0.1)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(angles[:, 0], angles[:, 1], angles[:, 2], alpha=0.1)
    plt.show()


def collect_gammas(N):
    n = 6
    x = np.arange(-N+1, N, dtype=int)
    xs = [x] * n
    # nth direct power of [-N, N] interval:
    a = np.meshgrid(*xs)
    # array of all possible [-N, N] intervals:
    b = np.stack(a, axis=-1).reshape((-1, n))
    # each re-ordered descending:
    b = - np.sort(- b, axis=-1)
    # keeping only the distinct ones:
    b = np.unique(b, axis=0)
    # keeping only the ones with nonnegative sums:
    b = b[b.sum(axis=1) >= 0]
    return b


gammas = collect_gammas(N=3)
print("gammas", gammas.shape)


# collect_cube_npys() ; exit()

# straight means haven't gone through canonization process
cs = np.load("all_straight_cubes.npy")

print("shrinking dataset")
cs = cs[:500]


gamma = np.zeros(6, dtype=int) ; gamma[0] += 1 ; gamma[-1] -= 1
gamma = np.array([1, 1, 1, -1, -1, -1], dtype=int)
# gamma = np.array([1, 0, 0,  0,  0, -1], dtype=int)


def evaluate_gamma_for_all(gamma, cs):
    gamma_perms, multipicity = collect_perms(gamma)

    power = cs[:, :, :, :, None] ** gamma_perms.T[None, :, None, None, :]
    # (3003, 6, 6, 6, 720) (cube_count, 0th axis that we do the prod over, 1th ax, 2nd ax, permutations)
    prod = np.prod(power, axis=1)

    # the * multipicity is because the unique removed multiplicities, now we put them back:
    summa = np.sum(prod, axis=-1) * multipicity

    summa = summa + np.conjugate(summa)
    assert np.allclose(summa.imag, 0)
    summa = summa.real
    # real array shaped (cube_count, 6, 6) for each 36 piercings of the cube_count cubes.
    return summa

    # print(summa.sum(axis=(1,2)).min())


dataset = []
for gamma in gammas:
    piercings = evaluate_gamma_for_all(gamma, cs)
    assert piercings.shape == (len(cs), 6, 6)
    dataset.append(piercings.flatten())

dataset = np.array(dataset)
print(dataset.shape)


u, s, vh = np.linalg.svd(dataset.T)
print(s.shape, s)
