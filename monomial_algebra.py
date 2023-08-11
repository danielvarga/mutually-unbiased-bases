import numpy as np

class Algebra:
    def __init__(self, coeffs, monoms):
        self.coeffs = coeffs
        self.monoms = monoms

    def __repr__(self):
        return f"Algebra({self.coeffs}, {self.monoms})"

    @staticmethod
    def zero(k):
        return Algebra(np.zeros((0, ), dtype=int), np.zeros((0, k), dtype=int))

    def __add__(self, other):
        coeffs1, monomials1 = self.coeffs, self.monoms
        coeffs2, monomials2 = other.coeffs, other.monoms

        all_monomials = np.concatenate((monomials1, monomials2))
        unique_monomials, unique_indices = np.unique(all_monomials, axis=0, return_index=True)

        combined_coeffs = np.zeros_like(unique_monomials[:, 0])
        for i in range(len(unique_monomials)):
            combined_coeffs[i] = np.sum(coeffs1[np.all(monomials1 == unique_monomials[i], axis=1)]) \
                + np.sum(coeffs2[np.all(monomials2 == unique_monomials[i], axis=1)])

        return Algebra(combined_coeffs, unique_monomials)

    def times_monomial(self, monom):
        return Algebra(self.coeffs, self.monoms + monom)

    def times_coeff(self, coeff):
        if coeff == 0:
            return Algebra.zero()
        else:
            return Algebra(self.coeffs * coeff, self.monoms)

    def __neg__(self):
        return Algebra(-self.coeffs, self.monoms)

    def __sub__(self, other):
        return self + (- other)

    def __mul__(self, other):
        coeffs, monomials = other.coeffs, other.monoms
        summa = Algebra.zero(monomials.shape[1])
        for coeff, monomial in zip(coeffs, monomials):
            new_monomial = self.times_coeff(coeff).times_monomial(monomial)
            summa = summa + new_monomial
        return summa


def test_algebra():
    one = Algebra(np.array([1, 2, 3]), np.array([[1, 2], [0, 1], [2, 3]]))
    other = Algebra(np.array([4, 5, 6]), np.array([[1, 2], [0, 1], [3, 4]]))
    print(one + other)
    print(one * other)


test_algebra() ; exit()
