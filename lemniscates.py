import numpy as np
from numpy import sqrt, conjugate
import matplotlib.pyplot as plt


n = 3600
I = 1j

true_delta = np.complex128(-0.12834049204788406+0.9917301639563593j)
true_alpha = np.complex128(0.8078018037463457+0.589454193185654j)
true_x =     np.complex128(0.5718790679708813+0.8203379374481935j)
true_d2_deg = np.array([-55.118405380635004, 60.00001971747099, 103.473320351059, 12.236700300504005, 127.35531748938601, -31.23708310199503])
true_d2 = np.exp(np.pi * 1j / 180 * true_d2_deg)
# or if you prefer:
assert np.allclose(true_d2, np.array([ 0.57188238-0.82033563j,  0.4999997 +0.86602558j, -0.23299256+0.97247852j,  0.97728033+0.21195083j, -0.60675613+0.79488805j,  0.8550288 -0.51858051j]))
true_W = 0.42593834


x = true_x
delta = np.exp(1j * np.linspace(0, 2 * np.pi, n))
# TODO do it the other way round, with fixing delta to its true value, and moving x.

# delta = true_delta

# this code is generated by supercanonize_mub.py:dump_solutions_in_python()
# see https://github.com/danielvarga/mutually-unbiased-bases/blob/dabcc32/supercanonize_mub.py if code rot sets in.

d21_numerator = conjugate(x)
d21_denominator = 1
d22_numerator = I*(sqrt(3) - I)
d22_denominator = 2
d23_numerator = 2*I*delta*conjugate(x) + sqrt(3)*conjugate(delta) - I*conjugate(delta) - conjugate(x) - sqrt(3)*I*conjugate(x)/3 + 1 + sqrt(3)*I/3
d23_denominator = sqrt(3)*delta - I*delta + sqrt(3)*conjugate(delta) + I*conjugate(delta) + 4*sqrt(3)*I/3
d24_numerator = -(sqrt(3)*delta/2 - 3*I*delta/2 + sqrt(3)*conjugate(delta) - 1 - sqrt(3)*I)*conjugate(x)
d24_denominator = sqrt(3)*delta/2 - 3*I*delta/2 + sqrt(3)*conjugate(delta) + 1 + sqrt(3)*I
d25_numerator = -2*sqrt(3)*delta - sqrt(3)*conjugate(delta) - 3*I*conjugate(delta) - 2 + 2*sqrt(3)*I
d25_denominator = sqrt(3)*delta - 3*I*delta + 2*sqrt(3)*conjugate(delta) + 2 + 2*sqrt(3)*I
d26_numerator = sqrt(3)*delta*conjugate(x) - 3*I*delta*conjugate(x) - 2*sqrt(3)*conjugate(delta) - conjugate(x) - sqrt(3)*I*conjugate(x) + 1 + sqrt(3)*I
d26_denominator = 2*sqrt(3)*delta + sqrt(3)*conjugate(delta) + 3*I*conjugate(delta) - 2 + 2*sqrt(3)*I
Walpha_numerator = I*(sqrt(3) + 3*I)*(conjugate(x) + 1)
Walpha_denominator = 6*sqrt(3)*delta + 3*sqrt(3)*conjugate(delta) + 9*I*conjugate(delta) - 6 + 6*sqrt(3)*I
# numerical values for debugging purposes:
true_d21_numerator = (0.5718790679708813-0.8203379374481935j)
true_d21_denominator = (1+0j)
true_d22_numerator = (1+1.7320508075688772j)
true_d22_denominator = (2+0j)
true_d23_numerator = (-2.604388406593981+0.958444405937476j)
true_d23_denominator = (1.5388758221220693+2.309401076758503j)
true_d24_numerator = (1.8793423650493297+1.4980575328954153j)
true_d24_denominator = (2.154156866591552+1.0656980299551897j)
true_d25_numerator = (-4.308313733183104+2.1313960599103794j)
true_d25_denominator = (4.308313733183104+2.1313960599103794j)
true_d26_numerator = (2.7511277035330868+3.941530199538587j)
true_d26_denominator = (0.30831373318310384+4.796807170365129j)
true_Walpha_numerator = (-3.2947702168761133+5.18358822142416j)
true_Walpha_denominator = (0.9249411995493115+14.390421511095388j)

dpi = 300

def show(v, c=None, s=2):
    if c is None:
        if isinstance(v, np.complex128) or len(v) == 1:
            c = "red"
        else:
            c = np.linspace(0, 1, len(v))
    plt.scatter(v.real, v.imag, c=c, s=s)


def title_or_latex(s):
    do_title = False
    if do_title:
        plt.title(s)
    else:
        print("\\caption{"+s+"}")
        print()


fig = plt.figure(figsize=(10, 5), dpi=dpi)
show(delta)
show(d23_numerator)
show(d23_denominator, s=4)
show(d23_numerator / d23_denominator)
show(true_d2[2], c="red", s=10) # one-off
show(true_d23_numerator / true_d23_denominator, c="pink", s=2)
title_or_latex('''Red dot is true value of $d_{23}$. Circle is tested $\\delta$, 
slightly ascending line is $d_{23}$ numerator, flat line is $d_{23}$ denominator,
$\infty$ shape is $d_{23}$. $x$ is set to its true value.''')
plt.gca().set_aspect('equal')
plt.savefig('lemniscate_d23.png', bbox_inches='tight')
plt.close()


plt.figure(figsize=(10, 5), dpi=dpi)
# show(delta)
show(d25_numerator)
show(d25_denominator, s=4)
show(d25_numerator / d25_denominator)
show(true_d2[4], c="red", s=10) # one-off
show(true_d25_numerator / true_d25_denominator, c="pink", s=2)
title_or_latex('''Red dot is true value of $d_{25}$. Tested $\\delta$ not shown.
Ascending line is $d_{25}$ numerator, descending line is $d_{25}$ denominator,
$d_{25}$ is C shape, a subset of the unit circle. $x$ is set to its true value.''')
plt.gca().set_aspect('equal')
plt.savefig('lemniscate_d25.png', bbox_inches='tight')
plt.close()


plt.figure(figsize=(10, 7), dpi=dpi)
show(delta)
show(d26_numerator)
show(d26_denominator, s=4)
show(d26_numerator / d26_denominator)
show(true_d2[5], c="red", s=10) # one-off
show(true_d26_numerator / true_d26_denominator, c="pink", s=5)
title_or_latex('''Red dot is true value of $d_{26}$. Circle is tested $\\delta$.
Lower line is $d_{26}$ numerator, higher line is $d_{26}$ denominator,
$\infty$ shape is $d_{26}$ itself. $x$ is set to its true value.''')
plt.gca().set_aspect('equal')
plt.savefig('lemniscate_d26.png', bbox_inches='tight')
plt.close()


plt.figure(figsize=(10, 10), dpi=dpi)
show(delta)
show(Walpha_numerator, c="green", s=10)
show(Walpha_denominator, s=4)
show(Walpha_numerator / Walpha_denominator)
show(delta * true_W, c="pink")
show(np.linspace(0, true_alpha * true_W, 1000), c="pink")
show(true_alpha * true_W, c="red", s=10)
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
# paper reader can't zoom out
# title_or_latex('''Red dot is true $W\\alpha$. Circle is tested $\\delta$, green dot is $W\\alpha$ numerator,
# line is $W\\alpha$ denominator (only seen when zoomed out), C shape is $W\\alpha$ itself.
# For reference, shown in pink the circle with radius $W$ and the line with direction $\\alpha$.
# They intersect at the true $W\\alpha$. $x$ is set to its true value.''')
title_or_latex('''Red dot is true $W\\alpha$. Circle is tested $\\delta$, green dot is $W\\alpha$ numerator.
$W\\alpha$ denominator is oscillator (line shape) outside the image. C shape is $W\\alpha$ itself.
For reference, the circle with radius $W$ and the line with direction $\\alpha$ are shown in pink.
They intersect at the true $W\\alpha$. $x$ is set to its true value.''')
plt.gca().set_aspect('equal')
plt.savefig('lemniscate_Walpha.png', bbox_inches='tight')
plt.close()
