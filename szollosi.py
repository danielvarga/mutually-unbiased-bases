import numpy as np
'''
F1[a_,b_,c_,d_,e_]:=a^2 b c d e+a b^2 c d e+b c^2 d e+b^2 c^2 d e+a c d^2 e+a^2 c d^2 e-a c d e^2-b c d e^2-a c^2 d e^2-a b c^2 d e^2-b c d^2 e^2-a b c d^2 e^2;
F2[a_,b_,c_,d_,e_]:=(-a b^2 c-a b^2 c^2-a^2 b d-a b c^2 d-a^2 b d^2-a b c d^2+a^2 b c e-b^2 c e+b c^2 e-a b^2 c^2 e-a^2 d e+a b^2 d e-a c^2 d e+b^2 c^2 d e+a d^2 e-a^2 b d^2 e+a^2 c d^2 e-b c d^2 e+a b c e^2+b c^2 e^2+a b d e^2+b c^2 d e^2+a d^2 e^2+a c d^2 e^2);
F3[a_,b_,c_,d_,e_]:=(a b c+a^2 b c+a b d+a b^2 d+a^2 b c d+a b^2 c d-b^2 c e-a b^2 c e-a^2 d e-a^2 b d e-a c d e-b c d e);
G1[a_,b_,c_,d_,e_]:=2 a b c d e+a^2 b c d e+a b^2 c d e+b c^2 d e+2 a b c^2 d e+b^2 c^2 d e+a c d^2 e+a^2 c d^2 e+2 a b c d^2 e+2 a b c d e^2+2 b c^2 d e^2+2 a c d^2 e^2;
G2[a_,b_,c_,d_,e_]:=(2 a b c d+2 a^2 b c d+2 a b^2 c d+2 a b c e+a^2 b c e+2 a b^2 c e+b c^2 e+2 a b c^2 e+2 b^2 c^2 e+2 a b d e+2 a^2 b d e+a b^2 d e+2 a c d e+2 a^2 c d e+2 b c d e+12 a b c d e+2 a^2 b c d e+2 b^2 c d e+2 a b^2 c d e+2 b c^2 d e+2 a b c^2 d e+b^2 c^2 d e+a d^2 e+2 a^2 d^2 e+2 a b d^2 e+2 a c d^2 e+a^2 c d^2 e+2 a b c d^2 e+a b c e^2+b c^2 e^2+a b d e^2+2 a c d e^2+2 b c d e^2+2 a b c d e^2+b c^2 d e^2+a d^2 e^2+a c d^2 e^2);
G3[a_,b_,c_,d_,e_]:=(a b c+a^2 b c+2 a b^2 c+a b d+2 a^2 b d+a b^2 d+2 a b c d+a^2 b c d+a b^2 c d+2 a b c e+2 a b d e+2 a b c d e);
F[a_,b_,c_,d_,e_]:=-(F3[a,b,c,d,e]G1[a,b,c,d,e]-F1[a,b,c,d,e]G3[a,b,c,d,e])/(F3[a,b,c,d,e]G2[a,b,c,d,e]-F2[a,b,c,d,e]G3[a,b,c,d,e]);
P[a_,b_,c_,d_,e_]:=a^4b^4c^3d^3e^3((F3[a,b,c,d,e]G1[a,b,c,d,e]-F1[a,b,c,d,e]G3[a,b,c,d,e])(F3[1/a,1/b,1/c,1/d,1/e]G1[1/a,1/b,1/c,1/d,1/e]-F1[1/a,1/b,1/c,1/d,1/e]G3[1/a,1/b,1/c,1/d,1/e])-(F3[a,b,c,d,e]G2[a,b,c,d,e]-F2[a,b,c,d,e]G3[a,b,c,d,e])(F3[1/a,1/b,1/c,1/d,1/e]G2[1/a,1/b,1/c,1/d,1/e]-F2[1/a,1/b,1/c,1/d,1/e]G3[1/a,1/b,1/c,1/d,1/e]));
darab=10;
Hads={};
r:=RandomInteger[{0,1000}];
While[Length[Hads]<darab,

{a,b,c,d}=N[Exp[2\[Pi] I/1000{r,r,r,r}],500];
solsB=#[[1,2]]&/@NSolve[(P[a,b,c,d,e]//Simplify)==0,e,100];
SOLB=Select[Subsets[solsB,{3}],Abs[N[1+a+b+(Plus@@#),100]]<0.001&];
SOLB[[1,1]];
{e1,e2,e3}=SOLB[[1]];
solsC=#[[1,2]]&/@NSolve[(P[a,c,b,d,e]//Simplify)==0,e,100];
SOLC=Select[Subsets[solsC,{3}],Abs[N[1+a+c+(Plus@@#),100]]<0.001&];
SOLC[[1,1]];
{g1,g2,g3}=SOLC[[1]];
{f1,f2,f3}=F[a,b,c,d,#]&/@{e1,e2,e3};
{h1,h2,h3}=F[a,c,b,d,#]&/@{g1,g2,g3};
A=(1    1    1
   1    a    b
   1    c    d
);
B=(1    1    1
   e1   e2   e3
   f1   f2   f3
);
C=(1   g1   h1
    1   g2   h2
    1   g3   h3
);
D=-cc.Transpose[1/A].Transpose[Conjugate[Inverse[B]]];

H=N[(
1       1       1       1              1              1
1       a       b       e1             e2             e3
1       c       d       f1             f2             f3
1       g1      h1      D[[1,1]]       D[[1,2]]       D[[1,3]]
1       g2      h2      D[[2,1]]       D[[2,2]]       D[[2,3]]
1       g3      h3      D[[3,1]]       D[[3,2]]       D[[3,3]]
)];
If[Plus@@Flatten[Abs[Abs[H]-1]]<0.00001,
If[Plus@@Flatten[Abs[(H.Transpose[Conjugate[H]]-6IdentityMatrix[6])]]<0.0001,
Hads=Join[Hads,{H}]]]
]
'''

# these 8 functions were automatically converted by
# cat szollosi.mathematica | tail -n +2 | head -8 | tr ' ' '*' | sed "s/\([2-9]\)\([a-e]\)/\1*\2/g" | tr '[]' '()' | sed "s/\()\)\([FG]\)/\1*\2/g" | sed "s/\^/**/g" | sed "s/)(/)*(/g" | sed "s/^/def /" | sed "s/=/@    return /g" | tr '@' '\n' | tr -d '_;' | sed "s/\(\*[2-9]\)(/\1*(/g"
def F1(a,b,c,d,e):
    return a**2*b*c*d*e+a*b**2*c*d*e+b*c**2*d*e+b**2*c**2*d*e+a*c*d**2*e+a**2*c*d**2*e-a*c*d*e**2-b*c*d*e**2-a*c**2*d*e**2-a*b*c**2*d*e**2-b*c*d**2*e**2-a*b*c*d**2*e**2
def F2(a,b,c,d,e):
    return (-a*b**2*c-a*b**2*c**2-a**2*b*d-a*b*c**2*d-a**2*b*d**2-a*b*c*d**2+a**2*b*c*e-b**2*c*e+b*c**2*e-a*b**2*c**2*e-a**2*d*e+a*b**2*d*e-a*c**2*d*e+b**2*c**2*d*e+a*d**2*e-a**2*b*d**2*e+a**2*c*d**2*e-b*c*d**2*e+a*b*c*e**2+b*c**2*e**2+a*b*d*e**2+b*c**2*d*e**2+a*d**2*e**2+a*c*d**2*e**2)
def F3(a,b,c,d,e):
    return (a*b*c+a**2*b*c+a*b*d+a*b**2*d+a**2*b*c*d+a*b**2*c*d-b**2*c*e-a*b**2*c*e-a**2*d*e-a**2*b*d*e-a*c*d*e-b*c*d*e)
def G1(a,b,c,d,e):
    return 2*a*b*c*d*e+a**2*b*c*d*e+a*b**2*c*d*e+b*c**2*d*e+2*a*b*c**2*d*e+b**2*c**2*d*e+a*c*d**2*e+a**2*c*d**2*e+2*a*b*c*d**2*e+2*a*b*c*d*e**2+2*b*c**2*d*e**2+2*a*c*d**2*e**2
def G2(a,b,c,d,e):
    return (2*a*b*c*d+2*a**2*b*c*d+2*a*b**2*c*d+2*a*b*c*e+a**2*b*c*e+2*a*b**2*c*e+b*c**2*e+2*a*b*c**2*e+2*b**2*c**2*e+2*a*b*d*e+2*a**2*b*d*e+a*b**2*d*e+2*a*c*d*e+2*a**2*c*d*e+2*b*c*d*e+12*a*b*c*d*e+2*a**2*b*c*d*e+2*b**2*c*d*e+2*a*b**2*c*d*e+2*b*c**2*d*e+2*a*b*c**2*d*e+b**2*c**2*d*e+a*d**2*e+2*a**2*d**2*e+2*a*b*d**2*e+2*a*c*d**2*e+a**2*c*d**2*e+2*a*b*c*d**2*e+a*b*c*e**2+b*c**2*e**2+a*b*d*e**2+2*a*c*d*e**2+2*b*c*d*e**2+2*a*b*c*d*e**2+b*c**2*d*e**2+a*d**2*e**2+a*c*d**2*e**2)
def G3(a,b,c,d,e):
    return (a*b*c+a**2*b*c+2*a*b**2*c+a*b*d+2*a**2*b*d+a*b**2*d+2*a*b*c*d+a**2*b*c*d+a*b**2*c*d+2*a*b*c*e+2*a*b*d*e+2*a*b*c*d*e)
def F(a,b,c,d,e):
    return -(F3(a,b,c,d,e)*G1(a,b,c,d,e)-F1(a,b,c,d,e)*G3(a,b,c,d,e))/(F3(a,b,c,d,e)*G2(a,b,c,d,e)-F2(a,b,c,d,e)*G3(a,b,c,d,e))
def P(a,b,c,d,e):
    return a**4*b**4*c**3*d**3*e**3*((F3(a,b,c,d,e)*G1(a,b,c,d,e)-F1(a,b,c,d,e)*G3(a,b,c,d,e))*(F3(1/a,1/b,1/c,1/d,1/e)*G1(1/a,1/b,1/c,1/d,1/e)-F1(1/a,1/b,1/c,1/d,1/e)*G3(1/a,1/b,1/c,1/d,1/e))-(F3(a,b,c,d,e)*G2(a,b,c,d,e)-F2(a,b,c,d,e)*G3(a,b,c,d,e))*(F3(1/a,1/b,1/c,1/d,1/e)*G2(1/a,1/b,1/c,1/d,1/e)-F2(1/a,1/b,1/c,1/d,1/e)*G3(1/a,1/b,1/c,1/d,1/e)))

args = (1,2,3,4,5)
for f in (F1, F2, F3, G1, G2, G3, F, P):
    print(f(*args))


def verify_hadamard(b, atol=1e-4):
    n = len(b)
    assert np.allclose(np.abs(b) ** 2, 1 / n, atol=atol)
    prod = np.conjugate(b.T) @ b
    assert np.allclose(prod, np.eye(n), atol=atol)


def random_phases(shape):
    return np.exp(2 * np.pi * 1j * np.random.uniform(size=shape))


def random_hadamard():
    ones = np.ones((3, 3), dtype=np.complex128)
    A = ones.copy()
    a, b, c, d = random_phases((4, ))
    A[1,1], A[1,2], A[2,1], A[2,2] = a, b, c, d

    '''
solsB=#[[1,2]]&/@NSolve[(P[a,b,c,d,e]//Simplify)==0,e,100];
SOLB=Select[Subsets[solsB,{3}],Abs[N[1+a+b+(Plus@@#),100]]<0.001&];
SOLB[[1,1]];
{e1,e2,e3}=SOLB[[1]];
solsC=#[[1,2]]&/@NSolve[(P[a,c,b,d,e]//Simplify)==0,e,100];
SOLC=Select[Subsets[solsC,{3}],Abs[N[1+a+c+(Plus@@#),100]]<0.001&];
SOLC[[1,1]];
{g1,g2,g3}=SOLC[[1]];
{f1,f2,f3}=F[a,b,c,d,#]&/@{e1,e2,e3};
{h1,h2,h3}=F[a,c,b,d,#]&/@{g1,g2,g3};
    '''

    # placeholder until the above is ported:
    e = random_phases((3, ))
    f = random_phases((3, ))
    g = random_phases((3, ))
    h = random_phases((3, ))

    B = ones.copy()
    B[1, :] = e
    B[2, :] = f

    C = ones.copy()
    C[:, 1] = g
    C[:, 2] = h

    D = -C @ np.transpose(1/A) @ np.transpose(np.conjugate(np.linalg.inv(B)))

    H = np.zeros((6, 6), dtype=np.complex128)
    H[:3, :3] = A
    H[:3, 3:] = B
    H[3:, :3] = C
    H[3:, 3:] = D
    return H


verify_hadamard(random_hadamard())
