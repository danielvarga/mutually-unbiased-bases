import numpy as np
import itertools


class Algebra:
    def __init__(self, coeffs, monoms):
        assert len(coeffs) == len(monoms)
        self.coeffs = np.array(coeffs)
        self.monoms = np.array(monoms)

    def __repr__(self):
        return f"Algebra({self.coeffs}, {self.monoms})"

    @property
    def shape(self):
        return self.monoms.shape[1:]

    @staticmethod
    def zero(shape):
        return Algebra(np.zeros((0, ), dtype=int), np.zeros([0] + list(shape), dtype=int))

    def __add__(self, other):
        assert self.shape == other.shape
        coeffs1, monomials1 = self.coeffs, self.monoms
        coeffs2, monomials2 = other.coeffs, other.monoms

        all_monomials = np.concatenate((monomials1, monomials2))
        unique_monomials, unique_indices = np.unique(all_monomials, axis=0, return_index=True)

        # dtype=coeffs2.dtype because we start with integer 0, but cast to float if we ever encounter a float coeff.
        # very flaky, sorry.
        combined_coeffs = np.zeros(len(unique_monomials), dtype=coeffs2.dtype)
        dimension = len(self.monoms.shape)
        axis = tuple(range(1, dimension))
        for i in range(len(unique_monomials)):
            combined_coeffs[i] = np.sum(coeffs1[np.all(monomials1 == unique_monomials[i], axis=axis)]) \
                + np.sum(coeffs2[np.all(monomials2 == unique_monomials[i], axis=axis)])

        return Algebra(combined_coeffs, unique_monomials)

    def times_monomial(self, monom):
        assert self.shape == monom.shape
        return Algebra(self.coeffs, self.monoms + monom[None, ...])

    def times_coeff(self, coeff):
        if coeff == 0:
            return Algebra.zero(self.shape)
        else:
            return Algebra(self.coeffs * coeff, self.monoms)

    def __neg__(self):
        return Algebra(-self.coeffs, self.monoms)

    def __sub__(self, other):
        assert self.shape == other.shape
        return self + (- other)

    def __mul__(self, other):
        assert self.shape == other.shape
        coeffs, monomials = other.coeffs, other.monoms
        summa = Algebra.zero(other.shape)
        for coeff, monomial in zip(coeffs, monomials):
            new_monomial = self.times_coeff(coeff).times_monomial(monomial)
            summa = summa + new_monomial
        return summa

    def conjugate(self):
        return Algebra(self.coeffs, -self.monoms)


def test_algebra():
    one = Algebra([1, 2, 3], [[1, 2], [0, 1], [2, 3]])
    other = Algebra([4, 5, 6], [[1, 2], [0, 1], [3, 4]])
    print(one + other)
    print(one * other)


# it's really not tensorial. it's the same thing, it's just that
# we allow variables to come from some x_{i,j,k}, not just x_{i}.
def test_tensorial_algebra():
    monomials = np.arange(12, dtype=int).reshape((3, 2, 2)) # that's 3 monomials of tensor shape (2, 2).
    one = Algebra([1, 2, 3], monomials)
    other = Algebra([4, 5, 6], monomials)
    print(one + other)
    print(one * other)


# test_algebra() ; exit()
# test_tensorial_algebra()


# just for future reference and maybe for later testing
def standalone_formula_3d(c, gamma):
    n = c.shape[0]
    assert c.shape == (n, n, n)
    assert gamma.shape == (n, )
    # first we do the powering on axis=1.
    # then we multiply over axis=1.
    # then we sum over axis=2.
    # then we take magnitude squared.
    # then we average over axis=0.
    powered = c ** gamma[None, :, None]
    vs = powered.prod(axis=1).sum(axis=1)
    values = np.abs(vs) ** 2
    return values.sum()


def standalone_formula_2d(h, gamma):
    n = h.shape[0]
    assert h.shape == (n, n)
    assert gamma.shape == (n, )
    # first we do the powering on axis=1.
    # then we multiply over axis=1.
    # then we sum over axis=2.
    # then we take magnitude squared.
    # then we average over axis=0.
    powered = h ** gamma[:, None]
    vs = powered.prod(axis=0).sum(axis=0)
    value = np.abs(vs) ** 2
    return value


def function_g(gamma):
    monomials = []
    for i in range(6):
        m = np.zeros((6, 6), dtype=int)
        m[:, i] = gamma
        monomials.append(m)
    monomials = np.array(monomials)
    sm = Algebra(np.ones(len(monomials), dtype=int), monomials)
    magnitude_squared = sm * sm.conjugate()
    return magnitude_squared


def test_function_g():
    gamma = np.array([1, 1, 1, -1, -1, -1])
    magnitude_squared = function_g(gamma)
    print(magnitude_squared)
    for i, coeff in enumerate(magnitude_squared.coeffs):
        if coeff == 6:
            print("the one with six", magnitude_squared.monoms[i])


def cube_g_with_fixed_perm(gamma, slc):
    monomials = []
    for i in range(6):
        m = np.zeros((6, 6, 6), dtype=int)
        m[slc, :, i] = gamma
        monomials.append(m)
    monomials = np.array(monomials)
    sm = Algebra(np.ones(len(monomials), dtype=int), monomials)
    magnitude_squared = sm * sm.conjugate()
    return magnitude_squared


def cube_g_with_given_perm(gamma, axis_perm, slc):
    g = cube_g_with_fixed_perm(gamma, slc)
    return Algebra(g.coeffs, np.transpose(g.monoms, [0] + [axis+1 for axis in axis_perm]))


def cube_g(gamma):
    summa = Algebra.zero((6, 6, 6))
    for axis_perm in itertools.permutations(range(3)):
        for slc in range(6):
            summa = summa + cube_g_with_given_perm(gamma, axis_perm, slc)
    return summa


gamma = np.array([1, 1, 1, -1, -1, -1])
g = cube_g(gamma)

print(np.unique(g.coeffs))
