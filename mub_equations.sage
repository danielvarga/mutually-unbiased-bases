import numpy as np

print("Hello!")


N = 6

# H and G are (6, 6) complex, represented as (2 (H or G), 6, 6, 2 (real or imag))


vars = np.zeros((2, N, N, 2), dtype=np.object)
matrices = np.zeros((2, N, N), dtype=np.object)
for matrix_id, matrix_name in enumerate(("G", "H")):
    for jj in range(N):
        for kk in range(N):
            for part_id, part_name in enumerate(("r", "i")):
                vars[matrix_id, jj, kk, part_id] = var(f"{matrix_name}_{jj+1}{kk+1}_{part_name}", domain=RR)
            matrices[matrix_id, jj, kk] = vars[matrix_id, jj, kk, 0] + i * vars[matrix_id, jj, kk, 1]


# this is not an (N, N) complex matrix, this is an (N, N, 2) array.
def create_unimodularity_constraints(m):
    unimodularity_constraints = [m[jj, kk, 0] ** 2 + m[jj, kk, 1] ** 2 - 1 for jj in range(N) for kk in range(N)]
    return unimodularity_constraints

unimodularity_constraints = create_unimodularity_constraints(vars[0]) + create_unimodularity_constraints(vars[1])

print("when we say real/complex something_constraints, real constraint means that by design, the imaginary part is zero.")

print(len(unimodularity_constraints), "real unimodularity_constraints", unimodularity_constraints[:3])


# unitarity next!

def scalar_product(v1, v2):
    return sum([conjugate(v1[ii]) * v2[ii] for ii in range(N)])


# this is an (N, N) complex matrix, not an (N, N, 2) array.
def create_unitarity_constraints(m):
    unitarity_constraints = []
    row_num, col_num = m.shape
    for jj in range(col_num):
        for kk in range(jj, col_num):
            sp = scalar_product(m[:, jj], m[:, kk])
            target = 1 if (jj == kk) else 0
            unitarity_constraints.append(sp - target)
    return unitarity_constraints


# orthogonality_constraints and unit_constraints together give Hadamardness.
unitarity_constraints = create_unitarity_constraints(matrices[0]) + create_unitarity_constraints(matrices[1])
print(len(unitarity_constraints), "complex unitarity_constraints", unitarity_constraints[:3])


# now unbiasedness!

def unbiased_vectors(v1, v2):
    assert v1.shape == v2.shape == (N, )
    sp = scalar_product(v1, v2)
    squared_magnitude = sp * conjugate(sp)
    return squared_magnitude - N


def create_unbiasedness_constraints(m1, m2):
    assert m1.shape == m2.shape == (N, N)
    row_num, col_num = m1.shape
    unbiasedness_constraints = []
    for jj in range(col_num):
        for kk in range(col_num):
            v1 = m1[:, jj]
            v2 = m2[:, kk]
            unbiasedness_constraints.append(unbiased_vectors(v1, v2))
    return unbiasedness_constraints


unbiasedness_constraints = create_unbiasedness_constraints(matrices[0], matrices[1])
print(len(unbiasedness_constraints), "real unbiasedness_constraints", unbiasedness_constraints[:3])

# now the parquet equations!

def half_conjugate(m):
    row_num, col_num = m.shape
    half = row_num // 2
    m2 = m.copy()
    for jj in range(half, row_num):
        m2[jj, :] = conjugate(m2[jj, :])
    return m2


def product_of_elements(v):
    return reduce((lambda x, y: x * y), v)


def nu(m, k):
    row_num, col_num = m.shape
    result = m[0, :].copy()
    m = half_conjugate(m)
    m = m ** k
    for kk in range(col_num):
        result[kk] = product_of_elements(m[:, kk])
    return result


def G(m, k):
    nu_vec = nu(m, k)
    sp = sum(nu_vec)
    return sp * conjugate(sp) / N / N


expand_all = np.vectorize(expand)

g = G(matrices[0], 1)


# and now let's plug in a concrete MUB to verify it up to numerical precision
