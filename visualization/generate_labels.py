# python generate_labels.py | sed "s/G/\\\gamma/g" | python generate_tikz.py > tmp.tex && pdflatex tmp && open tmp.pdf


import numpy as np
from sympy import *

from base import *


def circulant_sym(first_row):
    a, b, c = first_row
    return Matrix([[a, b, c], [c, a, b], [b, c, a]])


def szollosi_original_sym(first_row):
    a, b, c, d, e, f = first_row
    block1 = circulant_sym([a, b, c])
    block2 = circulant_sym([d, e, f])
    block3 = circulant_sym([conjugate(d), conjugate(f), conjugate(e)])
    block4 = circulant_sym([-conjugate(a), -conjugate(c), -conjugate(b)])
    blockcirculant = Matrix(BlockMatrix([[block1, block2], [block3, block4]]))
    return blockcirculant


def szollosi_modified_sym(first_row, gamma):
    blockcirculant = szollosi_original_sym(first_row)
    blockcirculant[3:, :] *= gamma
    return blockcirculant


INTERLACE = [0, 3, 1, 4, 2, 5]
DEINTERLACE = [0, 2, 4, 1, 3, 5]
MIRROR = [0, 1, 2, 5, 4, 3]
TRANSPOSITION = [1, 0, 3, 2, 5, 4]


def interlace(h):
    h = h[INTERLACE, :]
    return h[:, INTERLACE]


def deinterlace(h):
    h = h[DEINTERLACE, :]
    return h[:, DEINTERLACE]

def mirror(h):
    h = h[MIRROR, :]
    return h[:, MIRROR]


# sz_3x3 means it consists of 3x3 blocks (those are not Hadamard)
def convert_3x3_to_2x2(sz_3x3):
    return interlace(mirror(sz_3x3))


# sz_2x2 means it consists of 2x2 blocks (those are Hadamard)
def convert_2x2_to_3x3(sz_2x2):
    return mirror(deinterlace(sz_2x2))


# note: there's also an implementation base.py:conjugate_pair(sx) that operates on sz_3x3 matrices.
def conjugate_pair(sz):
    sz = conjugate(sz)
    sz = sz[TRANSPOSITION, :]
    sz = sz[:, TRANSPOSITION]
    return sz


def build_cube():
    first_rows = np.array([symbols("a b c d e f"), symbols("a' b' c' d' e' f'"), symbols("a'' b'' c'' d'' e'' f''")])
    gammas = np.array(symbols("G G' G''"))
    szs = [convert_3x3_to_2x2(szollosi_modified_sym(first_row, gamma)) for first_row, gamma in zip(first_rows, gammas)]
    cube = np.zeros((6, 6, 6), dtype=object)
    for i in range(3):
        cube[2 * i] = szs[i]
        cube[2 * i + 1] = conjugate_pair(szs[i])
    return cube, first_rows, gammas


def test():
    first_row = symbols("a b c d e f")
    gamma = symbols("gamma")
    sz_3x3 = szollosi_modified_sym(first_row, gamma)

    print("3x3 blocks:")
    print(sz_3x3)
    sz_2x2 = convert_3x3_to_2x2(sz_3x3)
    print("2x2 blocks after mirroring and interlacing:")
    print(sz_2x2)
    sz_3x3_again = convert_2x2_to_3x3(sz_2x2)
    assert np.all(np.array(sz_3x3_again - sz_3x3) == 0)

    sz_2x2_pair = conjugate_pair(sz_2x2)
    print("conjugate pair of 2x2 form:")
    print(latex(sz_2x2_pair))


# test() ; exit()


def dump_cube_labels_for_tikz():
    cube, first_rows, gammas = build_cube()

    for axis in range(3):
        axis_name = {0: 'z', 1: 'y', 2: 'x'}[axis]
        h = slic(cube, axis, 5 if axis==1 else 0)
        if axis != 0:
            h = h.T
        if axis == 1:
            h = h[::-1, :]
        for i in range(6):
            for j in range(6):
                element = h[i, j]
                print(f"{axis_name}\t{i + 1}\t{j + 1}\t${latex(element)}$")


dump_cube_labels_for_tikz() ; exit()
