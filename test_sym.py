import sympy as sym
from sympy.interactive import printing
printing.init_printing(use_latex=True)

def matrixfunction(A, B, C, t):
    mf = A - t * C * B.inv()
    return mf

def solvefor(A, B, t):
    C = sym.Matrix(3, 3, sym.symbols('C0:3(0:3)'))
    sol = sym.solve(matrixfunction(A, B, C, t), sym.symbols('C0:3(0:3)'))
    return sol

A = sym.Matrix([[1, 2, 3], [3, 2, 1], [2, 3, 1]])
B = sym.Matrix([[10, 2, 3], [30, 2, 1], [25, 3, 1]])

print(solvefor(A, B, 0.5))

from sympy.physics.quantum.dagger import Dagger

Cvars = sym.symbols('C0:3(0:3)')

C = sym.Matrix(3, 3, Cvars)

sol = sym.solve([C * C - sym.eye(3), Cvars[0] - 1, Cvars[4] - 1], Cvars)
print(sol)

sol = sym.solve([Dagger(C) * C - sym.eye(3), Cvars[0] - 1, Cvars[4] - 1, Cvars[8] - 1], Cvars)
print(sol)
