from sympy import *
from sympy.physics.quantum.dagger import Dagger


def enforce_norm_one(p, variables):
    for var in variables:
        p = p.subs(conjugate(var) * var, 1)
    return p


def conjugate_pair(sx):
    sxb = sx.copy()
    b00 = Dagger(sx[:3, :3])
    b01 = Dagger(sx[:3, 3:])
    b10 = Dagger(sx[3:, :3])
    b11 = Dagger(sx[3:, 3:])
    sxb[:3, :3] = b11
    sxb[:3, 3:] = b10
    sxb[3:, :3] = b01
    sxb[:3, :3] = b00
    return sxb


smalls = symbols('a, b, c, d, e, f')
a, b, c, d, e, f = smalls
bigs = symbols('A, B, C, D, E, F')
A, B, C, D, E, F = bigs
phi = symbols('phi')

variables = list(smalls) + [phi]

m = Matrix([
    [a, b, c, d, e, f],
    [c, a, b, f, d, e],
    [b, c, a, e, f, d],
    [A, B, C, D, E, F],
    [C, A, B, F, D, E],
    [B, C, A, E, F, D]
])

m[3:, :] *= phi

# bigvalues = [1/d, 1/f, 1/e, -1/a, -1/c, -1/b]
bigvalues = [conjugate(d), conjugate(f), conjugate(e), -conjugate(a), -conjugate(c), -conjugate(b)]

for bigvar, bigvalue in zip(bigs, bigvalues):
    m = m.subs(bigvar, bigvalue)


print(m)

def verify_unitarity(m, constraints):
    mdm = Dagger(m) @ m
    mdm = enforce_norm_one(mdm, variables)
    for constraint in constraints:
        mdm = mdm.subs(constraint, 0).subs(expand(conjugate(constraint)), 0)
    print(mdm)
    assert mdm == 6 * eye(6)
    return mdm


# this guarantees that m is unitary:
m_constraints = []
m_constraints.append(a*conjugate(c) + b*conjugate(a) + c*conjugate(b) + d*conjugate(f) + e*conjugate(d) + f*conjugate(e))
verify_unitarity(m, m_constraints)


# these guarantee that conjugate_pair(m) is unitary:
m_dual_constraints = m_constraints[:]
m_dual_constraints.append(a*d*conjugate(phi) + b*f*conjugate(phi) + c*e*conjugate(phi) - d*phi*conjugate(a) - e*phi*conjugate(b) - f*phi*conjugate(c))
m_dual_constraints.append(a*e*conjugate(phi) + b*d*conjugate(phi) + c*f*conjugate(phi) - d*phi*conjugate(c) - e*phi*conjugate(a) - f*phi*conjugate(b))
m_dual_constraints.append(a*f*conjugate(phi) + b*e*conjugate(phi) + c*d*conjugate(phi) - d*phi*conjugate(b) - e*phi*conjugate(c) - f*phi*conjugate(a))

m_dual = conjugate_pair(m)
verify_unitarity(m_dual, m_dual_constraints)
