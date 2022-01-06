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
W = 0.42593834


x = true_x
delta = np.exp(1j * np.linspace(0, 2 * np.pi, n))
# TODO do it the other way round, with fixing delta to its true value, and moving x.

# delta = true_delta

d21_numerator = conjugate(x)
true_d21_numerator = (0.5718790679708813-0.8203379374481935j)
d21_denominator = 1
true_d21_denominator = (1+0j)
d22_numerator = I*(sqrt(3) - I)
true_d22_numerator = (1+1.7320508075688772j)
d22_denominator = 2
true_d22_denominator = (2+0j)
d23_numerator = 2*I*delta*conjugate(x) + sqrt(3)*conjugate(delta) - I*conjugate(delta) - conjugate(x) - sqrt(3)*I*conjugate(x)/3 + 1 + sqrt(3)*I/3
true_d23_numerator = (-2.604388406593981+0.958444405937476j)
d23_denominator = sqrt(3)*delta - I*delta + sqrt(3)*conjugate(delta) + I*conjugate(delta) + 4*sqrt(3)*I/3
true_d23_denominator = (1.5388758221220693+2.309401076758503j)
d24_numerator = -(sqrt(3)*delta/2 - 3*I*delta/2 + sqrt(3)*conjugate(delta) - 1 - sqrt(3)*I)*conjugate(x)
true_d24_numerator = (1.8793423650493297+1.4980575328954153j)
d24_denominator = sqrt(3)*delta/2 - 3*I*delta/2 + sqrt(3)*conjugate(delta) + 1 + sqrt(3)*I
true_d24_denominator = (2.154156866591552+1.0656980299551897j)
d25_numerator = -2*sqrt(3)*delta - sqrt(3)*conjugate(delta) - 3*I*conjugate(delta) - 2 + 2*sqrt(3)*I
true_d25_numerator = (-4.308313733183104+2.1313960599103794j)
d25_denominator = sqrt(3)*delta - 3*I*delta + 2*sqrt(3)*conjugate(delta) + 2 + 2*sqrt(3)*I
true_d25_denominator = (4.308313733183104+2.1313960599103794j)
d26_numerator = sqrt(3)*delta*conjugate(x) - 3*I*delta*conjugate(x) - 2*sqrt(3)*conjugate(delta) - conjugate(x) - sqrt(3)*I*conjugate(x) + 1 + sqrt(3)*I
true_d26_numerator = (2.7511277035330868+3.941530199538587j)
d26_denominator = 2*sqrt(3)*delta + sqrt(3)*conjugate(delta) + 3*I*conjugate(delta) - 2 + 2*sqrt(3)*I
true_d26_denominator = (0.30831373318310384+4.796807170365129j)


'''
# it's really -6Walpha, not W, sorry.
W_numerator = -2*I*(sqrt(3) + 3*I)*(conjugate(x) + 1)[np.newaxis]
W_denominator = (2*sqrt(3)*delta + sqrt(3)*conjugate(delta) + 3*I*conjugate(delta) - 2 + 2*sqrt(3)*I)
'''


def show(v, c=None, s=2):
    if c is None:
        if isinstance(v, np.complex128) or len(v) == 1:
            c = "red"
        else:
            c = np.linspace(0, 1, len(v))
    plt.scatter(v.real, v.imag, c=c, s=s)


show(delta)
show(d23_numerator)
show(d23_denominator)
show(d23_numerator / d23_denominator)
show(true_d2[2], c="red", s=10) # one-off
show(true_d23_numerator / true_d23_denominator, c="pink", s=5)
plt.title('''circle is tested $\\delta$, red dot is true $\\delta$.
flat line is $d_{23}$ denominator, sloped line is $d_{23}$ numerator,
$\infty$ is $d_{23}$ itself. x is set to its true value.''')
plt.gca().set_aspect('equal')
plt.show()

# show(delta)
show(d25_numerator)
show(d25_denominator)
show(d25_numerator / d25_denominator)
show(true_d2[4], c="red", s=10) # one-off
show(true_d25_numerator / true_d25_denominator, c="pink", s=5)
plt.title('''tested $\\delta$ not shown, red dot is true $\delta$.
descending line is $d_{25}$ denominator, ascending line is $d_{25}$ numerator,
$d_{25}$ is C-shape, a subset of the unit circle. x is set to its true value.''')
plt.gca().set_aspect('equal')
plt.show()

show(delta)
show(d26_numerator)
show(d26_denominator)
show(d26_numerator / d26_denominator)
show(true_d2[5], c="red", s=10) # one-off
show(true_d26_numerator / true_d26_denominator, c="pink", s=5)
plt.title('''circle is tested $\\delta$, red dot is true $\\delta$.
gently ascending line is $d_{26}$ denominator, steep line is $d_{26}$ numerator,
$\infty$ is $d_{26}$ itself. x is set to its true value.''')
plt.gca().set_aspect('equal')
plt.show()

show(delta)
show(W_numerator / -6, c="green")
show(W_denominator)
show(W_numerator / W_denominator / -6)
show(delta * W, c="pink")
show(np.linspace(0, true_delta * W, 1000), c="pink")
show(true_delta * W, c="red", s=10)
plt.title('''circle is $\\delta$, red dot is true $W\\alpha$.
line is $W\\alpha$ denominator, green dot is $W\\alpha$ numerator, C-shape is $W\\alpha$ itself.
for reference, we also show the circle with radius $W$ and the line with direction $\\alpha$ in pink.
they intersect at the red dot. x is set to its true value.''')
plt.gca().set_aspect('equal')
plt.show()

exit()


show(delta)
show(d25_numerator)
show(d25_denominator)
show(d26_numerator)
show(d26_denominator)
delta = np.array([-0.12834049204788406+0.9917301639563593j])
show(delta)

plt.show()

'''
delta = np.linspace(0, 10, n)
d25 = -(2*sqrt(3)*delta + sqrt(3)*conjugate(delta) + 3*I*conjugate(delta) + 2 - 2*sqrt(3)*I)/(2*(sqrt(3)*delta/2 - 3*I*delta/2 + sqrt(3)*conjugate(delta) + 1 + sqrt(3)*I))
plt.plot(delta, np.abs(d25))
plt.show()
'''