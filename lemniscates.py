import numpy as np
from numpy import sqrt, conjugate
import matplotlib.pyplot as plt

n = 3600
I = 1j

true_delta = -0.12834049204788406+0.9917301639563593j
true_alpha = 0.8078018037463457+0.589454193185654j
true_x = np.exp(np.pi/180j * 55.118405380635004)
true_d2_deg = np.array([-55.118405380635004, 60.00001971747099, 103.473320351059, 12.236700300504005, 127.35531748938601, -31.23708310199503])
true_d2 = np.exp(np.pi/180j * true_d2_deg)
W = 0.42593834

x = true_x
delta = np.exp(1j * np.linspace(0, 2 * np.pi, n))
# TODO do it the other way round, with fixing delta to its true value, and moving x.

d21 = conjugate(x)[np.newaxis] # does not depend on delta, hence newaxis.

d22 = (I*(sqrt(3) - I)/2)[np.newaxis]

d23_numerator = (2*I*delta*conjugate(x) + sqrt(3)*conjugate(delta) - I*conjugate(delta) - conjugate(x) - sqrt(3)*I*conjugate(x)/3 + 1 + sqrt(3)*I/3)
d23_denominator = (sqrt(3)*delta - I*delta + sqrt(3)*conjugate(delta) + I*conjugate(delta) + 4*sqrt(3)*I/3)

# that's really just a rotated d23:
d24_numerator = -(sqrt(3)*delta/2 - 3*I*delta/2 + sqrt(3)*conjugate(delta) - 1 - sqrt(3)*I)*conjugate(x)
d24_denominator = (sqrt(3)*delta/2 - 3*I*delta/2 + sqrt(3)*conjugate(delta) + 1 + sqrt(3)*I)

d25_numerator = -(2*sqrt(3)*delta + sqrt(3)*conjugate(delta) + 3*I*conjugate(delta) + 2 - 2*sqrt(3)*I)
d25_denominator =  (2*(sqrt(3)*delta/2 - 3*I*delta/2 + sqrt(3)*conjugate(delta) + 1 + sqrt(3)*I))

d26_numerator = -2*(-sqrt(3)*delta*conjugate(x)/2 + 3*I*delta*conjugate(x)/2 + sqrt(3)*conjugate(delta) + conjugate(x)/2 + sqrt(3)*I*conjugate(x)/2 - 1/2 - sqrt(3)*I/2)
d26_denominator = (2*sqrt(3)*delta + sqrt(3)*conjugate(delta) + 3*I*conjugate(delta) - 2 + 2*sqrt(3)*I)

# it's really -6Walpha, not W, sorry.
W_numerator = -2*I*(sqrt(3) + 3*I)*(conjugate(x) + 1)[np.newaxis]
W_denominator = (2*sqrt(3)*delta + sqrt(3)*conjugate(delta) + 3*I*conjugate(delta) - 2 + 2*sqrt(3)*I)

def show(v, c=None):
    if c is None:
        if isinstance(v, np.complex128) or len(v) == 1:
            c = "red"
        else:
            c = np.linspace(0, 1, len(v))
    plt.scatter(v.real, v.imag, c=c, s=1)


show(delta)
show(d23_numerator)
show(d23_denominator)
show(d23_numerator / d23_denominator)
show(true_d2[2], c="red") # one-off
plt.title('''circle is tested $\\delta$, red dot is true $\\delta$.
flat line is $d_{23}$ denominator, sloped line is $d_{23}$ numerator,
$\infty$ is $d_{23}$ itself. x is set to its true value.''')
plt.gca().set_aspect('equal')
plt.show()

# show(delta)
show(d25_numerator)
show(d25_denominator)
show(d25_numerator / d25_denominator)
show(true_d2[4], c="red") # one-off
plt.title('''tested $\\delta$ not shown, red dot is true $\delta$.
descending line is $d_{25}$ denominator, ascending line is $d_{25}$ numerator,
$d_{25}$ is C-shape, a subset of the unit circle. x is set to its true value.''')
plt.gca().set_aspect('equal')
plt.show()

show(delta)
show(d26_numerator)
show(d26_denominator)
show(d26_numerator / d26_denominator)
show(true_d2[5], c="red") # one-off
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
show(true_delta * W, c="red")
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