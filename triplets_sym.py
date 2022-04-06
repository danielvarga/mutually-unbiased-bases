from sympy import *
from sympy.physics.quantum.dagger import Dagger


def enforce_norm_one(p, variables):
    for var in variables:
        p = p.subs(conjugate(var) * var, 1)
    return p


a, b, c, d, e, f = symbols('a, b, c, d, e, f')
A, B, C, D, E, F = symbols('A, B, C, D, E, F')
variables = [a, b, c, d, e, f] + [A, B, C, D, E, F]

m = Matrix([
    [a, b, c, d, e, f],
    [c, a, b, f, d, e],
    [b, c, a, e, f, d],
    [A, B, C, D, E, F],
    [C, A, B, F, D, E],
    [B, C, A, E, F, D]
])

print(m)

mdm = Dagger(m) @ m
mdm = enforce_norm_one(mdm, variables)

cons = []
cons.append(A*conjugate(C) + B*conjugate(A) + C*conjugate(B) + a*conjugate(c) + b*conjugate(a) + c*conjugate(b))
cons.append(D*conjugate(A) + E*conjugate(B) + F*conjugate(C) + d*conjugate(a) + e*conjugate(b) + f*conjugate(c))
cons.append(D*conjugate(C) + E*conjugate(A) + F*conjugate(B) + d*conjugate(c) + e*conjugate(a) + f*conjugate(b))
cons.append(D*conjugate(B) + E*conjugate(C) + F*conjugate(A) + d*conjugate(b) + e*conjugate(c) + f*conjugate(a))
cons.append(D*conjugate(F) + E*conjugate(D) + F*conjugate(E) + d*conjugate(f) + e*conjugate(d) + f*conjugate(e))

for con in cons:
    mdm = mdm.subs(con, 0).subs(expand(conjugate(con)), 0)

print(mdm)


