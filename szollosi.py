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


def verify_hadamard(b, atol=1e-4):
    n = len(b)
    assert np.allclose(np.abs(b) ** 2, 1, atol=atol)
    prod = np.conjugate(b.T) @ b
    assert np.allclose(prod, n * np.eye(n), atol=atol)


def random_phases(shape):
    return np.exp(2 * np.pi * 1j * np.random.uniform(size=shape))



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


def test_sympy():
    import sympy
    a, b, c, d, e = sympy.symbols('a b c d e')
    for f in (F1, F2, F3, G1, G2, G3, F, P):
        print("======")
        print(f(a,b,c,d,e))


def solve_p_for_e(abcd):
    import sympy
    a, b, c, d = abcd
    e = sympy.symbols('e')
    e_poly = sympy.expand(P(a, b, c, d, e))
    e_poly = sympy.Poly(e_poly, e)
    coeffs = e_poly.all_coeffs()
    coeffs = np.array([ np.complex128(coeff) for coeff in coeffs ])
    solutions = np.polynomial.polynomial.polyroots(coeffs)
    return solutions


def test_solve():
    import matplotlib.pyplot as plt
    r = 4
    n = 100
    x = np.linspace(-r, r, n)
    y = np.linspace(-r, r, n)
    xv, yv = np.meshgrid(x, y)
    e = xv + 1j * yv
    for i in range(10):
        np.random.seed(i)
        abcd = random_phases((4, ))
        a, b, c, d = abcd
        p = P(a, b, c, d, e)
        plt.imshow(-np.log(np.abs(p))[::-1, :])
        solutions = solve_p_for_e(abcd)
        solutions_proj = (solutions + r + r * 1j) / 2 / r * (n - 1)
        plt.scatter(solutions_proj.real, solutions_proj.imag)
        plt.show()
    abcd = random_phases((4, ))
    print(len(solutions), solutions)

# test_solve() ; exit()


def heatmap_e():
    import matplotlib.pyplot as plt
    r = 4
    n = 100
    x = np.linspace(-r, r, n)
    y = np.linspace(-r, r, n)
    xv, yv = np.meshgrid(x, y)
    e = xv + 1j * yv
    for i in range(20):
        np.random.seed(i)
        a, b, c, d = random_phases((4, ))
        p = P(a, b, c, d, e)
        plt.imshow(-np.log(np.abs(p)))
        plt.show()


# heatmap_e() ; exit()


def random_hadamard(seed=None):
    np.random.seed(seed)
    ones = np.ones((3, 3), dtype=np.complex128)
    A = ones.copy()
    abcd = random_phases((4, ))

    # abcd = np.exp(np.linspace(0.2, 0.8, 4) * np.pi * 2j)
    # abcdi = np.array([119, 515, 636, 943]) / 1000 ; abcd = np.exp(abcdi * np.pi * 2j)

    a, b, c, d = abcd
    acbd = a, c, b, d
    A[1,1], A[1,2], A[2,1], A[2,2] = a, b, c, d

    solutions_abcd = np.conjugate(solve_p_for_e(abcd))
    solutions_acbd = np.conjugate(solve_p_for_e(acbd))

    '''
    print("abcd", solutions_abcd)
    print("acbd", solutions_acbd)
    '''

    '''
    print(solutions_abcd)
    import matplotlib.pyplot as plt
    plt.scatter(solutions_abcd.real, solutions_abcd.imag)
    plt.show()
    '''

    import itertools
    for triple_abcd in itertools.combinations(solutions_abcd, 3):
        sb = 1 + a + b + sum(triple_abcd)
        if np.abs(sb) < 1e-3:
            break
        else:
            triple_abcd = None

    for triple_acbd in itertools.combinations(solutions_acbd, 3):
        sc = 1 + a + c + sum(triple_acbd)
        if np.abs(sc) < 1e-3:
            break
        else:
            triple_acbd = None

    if triple_abcd is None or triple_acbd is None:
        return None

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

    e = np.array(triple_abcd)
    g = np.array(triple_acbd)

    f = F(a, b, c, d, e)
    h = F(a, c, b, d, g)

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


for seed in range(100):
    print("====")
    H = random_hadamard(seed=seed)
    np.set_printoptions(precision=6, suppress=False, linewidth=100000)
    if H is not None:
        try:
            verify_hadamard(H)
            print(H)
        except:
            print("nope", seed)

