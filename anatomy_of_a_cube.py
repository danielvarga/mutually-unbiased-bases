from sympy import *
from sympy.physics.quantum.dagger import Dagger
from sympy.tensor.array import MutableDenseNDimArray


def circulant_sym(firstrow):
    a, b, c = firstrow
    return Matrix([[a, b, c], [c, a, b], [b, c, a]])


def szollosi_original_sym(firstrow):
    a, b, c, d, e, f = firstrow
    block1 = circulant_sym([a, b, c])
    block2 = circulant_sym([d, e, f])
    # block3 = circulant_sym([conjugate(d), conjugate(f), conjugate(e)])
    # block4 = circulant_sym([-conjugate(a), -conjugate(c), -conjugate(b)])
    block3 = circulant_sym([1 / d, 1 / f, 1 / e])
    block4 = circulant_sym([-1 / a, -1 / c, -1 / b])
    blockcirculant = Matrix(BlockMatrix([[block1, block2], [block3, block4]]))
    return blockcirculant


def szollosi_modified_sym(firstrow, phi):
    blockcirculant = szollosi_original_sym(firstrow)
    blockcirculant[3:, :] *= phi
    return blockcirculant


# Akos does not like to see conjugates.
def mydagger(m):
    return m.applyfunc(lambda x: 1 / x).T


def conjugate_pair_sym(sx):
    sxb = Matrix(sx.copy())
    b00 = mydagger(sx[:3, :3])
    b01 = mydagger(sx[:3, 3:])
    b10 = mydagger(sx[3:, :3])
    b11 = mydagger(sx[3:, 3:])
    sxb[:3, :3] = b11
    sxb[:3, 3:] = b10
    sxb[3:, :3] = b01
    sxb[:3, :3] = b00
    return sxb


def build_cube_from_slicepair_data(slicepair_data):
    c = MutableDenseNDimArray([0] * 216, (6, 6, 6))
    for i in range(3):
        firstrow, phi = slicepair_data[i]
        sx0_sym = szollosi_modified_sym(firstrow, phi)
        sx1_sym = conjugate_pair_sym(sx0_sym)
        c[2 * i, :, :] = sx0_sym
        c[2 * i + 1, :, :] = sx1_sym
    return ImmutableDenseNDimArray(c)


def create_symbolic_cube():
    # sy as in y-directional slice, a Fourier matrix.
    sy_sym = Matrix([[symbols(f'f_{i+1}{j+1}') for j in range(6)] for i in range(6)])
    phis = symbols('x_1 x_2 x_3') # Akos prefers x
    variables = [symbols(f'f_{i+1}{j+1}') for j in range(6) for i in range(6)] + list(phis)


    slicepair_data_sym = [(sy_sym[2 * i, :], phis[i]) for i in range(3)]
    c_sym = build_cube_from_slicepair_data(slicepair_data_sym)
    return c_sym, sy_sym, phis, variables, slicepair_data_sym


c_sym, sy_sym, phis, variables, slicepair_data_sym = create_symbolic_cube()
print("A Hadamard cube, parameterized by 3 rows of a Fourier slice, and 3 corresponding phi parameters")
print(c_sym)

print("\n====================\n")

# turns the 2x2x(3x3) block structure of Szollosi into 3x3x(2x2).
def interlace(b_orig):
    b = b_orig[:, [0, 3, 2, 4, 1, 5]]
    b = b[[0, 3, 2, 4, 1, 5], :]
    return b


firstrow = symbols("a b c d e f")
phi = symbols("x") # Akos prefers x

sx0 = szollosi_original_sym(firstrow)


def pretty(sx):
    for i in range(6):
        print(sx[i:i+1, :])


def header(s):
    print("--------")
    print(s)

header("straight Szollosi")
pretty(sx0)

header("conjugate pair of straight Szollosi")
sx1 = conjugate_pair_sym(sx0)
pretty(sx1)

sx0i = interlace(sx0)
header("straight Szollosi interlaced")
pretty(sx0i)

# modified means that the phi parameter is applied to 3 rows
sx0 = szollosi_modified_sym(firstrow, phi)
header("modified Szollosi")
pretty(sx0)

sx0i = interlace(sx0)
header("modified Szollosi interlaced")
pretty(sx0i)

header("interlacing the conjugate pair of the modified Szollosi")
sx1i = interlace(conjugate_pair_sym(sx0))
pretty(sx1i)

