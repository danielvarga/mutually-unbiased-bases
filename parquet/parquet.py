import numpy as np
from pulp import *

n = 5
N = 8

x = np.arange(-N+1, N, dtype=int)
xs = [x] * n
# nth direct power of [-N, N] interval:
a = np.meshgrid(*xs)
# array of all possible [-N, N] intervals:
b = np.stack(a, axis=-1).reshape((-1, n))
# ...that sum to 0:
b = b[b.sum(axis=-1) == 0]
# each re-ordered descending:
b = - np.sort(- b, axis=-1)
# keeping only the distinct ones:
b = np.unique(b, axis=0)
variables = set(tuple(var) for var in b)

lp = LpProblem("parquet", LpMinimize)
# for each element of variable, we create an LP variable bound between 0 and 1:
vardict = LpVariable.dicts("x", list(variables), lowBound=0, upBound=1)

# we set to one the all-0 variable to 1:
zero = vardict[tuple([0]*n)]
lp += zero == 1

# parquet constraints.
# let e_i be the ith basis vector.
# for each variable x = x_1 >= ... >= x_n, we set y=x-e_1, and constrain
# v(y+e_1) + ... + v(y+e_n) = 0
for var in variables:
    centre = np.array(var)
    centre[0] -= 1
    available = True
    parq_vars = []
    for i in range(n):
        pos = centre.copy()
        pos[i] += 1
        var_i = tuple(sorted(pos, reverse=True))
        if var_i not in variables:
            available = False
            break
        parq_vars.append(vardict[var_i])
    if available:
        lp += sum(parq_vars) == 1


def build(*v):
    return vardict[tuple(sorted(list(v) + [0] * (n - len(v)), reverse=True))]

# we'd like to know the minimum possible value of the following variables
# or sum of variables:
target = build(1, 1, -1, -1)
target = build(n, -n)
lp += target

target1 = build(2, -1, -1)
target2 = build(1, 1, -1, -1)
# lp += target1 + target2


lp.solve()
print(target, ">=", lp.objective.value())
