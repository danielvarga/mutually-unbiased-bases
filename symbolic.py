from sympy import symbols, MatrixSymbol, Matrix, eye
from sympy import solveset, solve, Eq
from sympy.physics.quantum.dagger import Dagger

n = 6

def varset(v, n):
    return f"{v}0:{n}(0:{n})"

Xvars = symbols(varset("X", n))
Yvars = symbols(varset("Y", n))
Zvars = symbols(varset("Z", n))
allvars = list(Xvars + Yvars + Zvars)
X = Matrix(n, n, Xvars)
Y = Matrix(n, n, Yvars)
Z = Matrix(n, n, Zvars)

'''
X = MatrixSymbol('X', n, n)
Y = MatrixSymbol('Y', n, n)
Z = MatrixSymbol('Z', n, n)
'''

a, b, c = symbols('a b c')

ux = Dagger(X) * X - eye(n)
uy = Dagger(Y) * Y - eye(n)
uz = Dagger(Z) * Z - eye(n)
# ux, uy, uz = [f.as_explicit() for f in (ux, uy, uz)]
s = solveset(ux, allvars) # , allvars)
print(s)
