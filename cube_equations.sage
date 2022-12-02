import numpy as np

print("Hello!")


def slic(c, direction, coord):
    if direction == 0:
        return c[coord, :, :]
    elif direction == 1:
        # T to be consistent with pierc direction 1:
        return c[:, coord, :].T
    elif direction == 2:
        return c[:, :, coord]
    assert False


def pierc(c, direction, coord1, coord2):
    if direction == 0:
        return c[:, coord1, coord2]
    elif direction == 1:
        # that's a decision right there! not c[coord1, :, coord2]
        return c[coord2, :, coord1]
    elif direction == 2:
        return c[coord1, coord2, :]
    assert False


N = 6


vars = np.zeros((N, N, N, 2), dtype=np.object)
cube = np.zeros((N, N, N), dtype=np.object)
for ii in range(N):
    for jj in range(N):
        for kk in range(N):
            for ll, name in enumerate(("x", "y")):
                vars[ii, jj, kk, ll] = var(f"{name}_{ii}{jj}{kk}", domain=RR)
            cube[ii, jj, kk] = vars[ii, jj, kk, 0] + i * vars[ii, jj, kk, 1]

unit_constraints = [vars[ii, jj, kk, 0] ** 2 + vars[ii, jj, kk, 1] ** 2 - 1 for ii in range(N) for jj in range(N) for kk in range(N)]
print(unit_constraints[:3])

piercing_constraints = []
for direction in range(3):
    for ii in range(N):
        for jj in range(N):
            line = pierc(cube, direction, ii, jj)
            piercing_constraints.append(sum(line) - 1)

print(piercing_constraints[:3])


def orthogonal(v1, v2):
    return sum([v1[ii] * conjugate(v2[ii]) for ii in range(N)])


# orthogonality_constraints and unit_constraints together give Hadamardness.
# thanks to the parallel axiom, we only need to ensure this for the three slic(c, direction, 0) slices.
orthogonality_constraints = []
for direction in range(3):
    plane = slic(cube, direction, 0)
    for ii in range(N):
        for jj in range(ii, N):
            orthogonality_constraints.append(orthogonal(plane[ii, :], plane[jj, :]))

print(orthogonality_constraints[:3])

# a way to guarantee the parallel axiom
checkerboard_constraints = []
