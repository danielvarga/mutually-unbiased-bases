import numpy as np
from numpy import sqrt, conjugate
import matplotlib.pyplot as plt

n = 3600
delta = np.exp(1j * np.linspace(0, 2 * np.pi, n))
I = 1j

true_delta = -0.12834049204788406+0.9917301639563593j
x = np.exp(np.pi/180j * 55.118)
# TODO do it the other way round, with fixing delta to its true value, and moving x.


p10 = conjugate(x)[np.newaxis]
p11 = (I*(sqrt(3) - I)/2)[np.newaxis]
p12_numerator = (2*I*delta*conjugate(x) + sqrt(3)*conjugate(delta) - I*conjugate(delta) - conjugate(x) - sqrt(3)*I*conjugate(x)/3 + 1 + sqrt(3)*I/3)
p12_denominator = (sqrt(3)*delta - I*delta + sqrt(3)*conjugate(delta) + I*conjugate(delta) + 4*sqrt(3)*I/3)
p14_numerator = -(2*sqrt(3)*delta + sqrt(3)*conjugate(delta) + 3*I*conjugate(delta) + 2 - 2*sqrt(3)*I)
p14_denominator =  (2*(sqrt(3)*delta/2 - 3*I*delta/2 + sqrt(3)*conjugate(delta) + 1 + sqrt(3)*I))
p15_numerator = -2*(-sqrt(3)*delta*conjugate(x)/2 + 3*I*delta*conjugate(x)/2 + sqrt(3)*conjugate(delta) + conjugate(x)/2 + sqrt(3)*I*conjugate(x)/2 - 1/2 - sqrt(3)*I/2)
p15_denominator = (2*sqrt(3)*delta + sqrt(3)*conjugate(delta) + 3*I*conjugate(delta) - 2 + 2*sqrt(3)*I)
# it's really -6Aalpha, not A, sorry.
A_numerator = -2*I*(sqrt(3) + 3*I)*(conjugate(x) + 1)[np.newaxis] # does not depend on delta
A_denominator = (2*sqrt(3)*delta + sqrt(3)*conjugate(delta) + 3*I*conjugate(delta) - 2 + 2*sqrt(3)*I)

def show(v):
    if len(v) > 1:
        c = np.linspace(0, 1, len(v))
    else:
        c = "red"
    plt.scatter(v.real, v.imag, c=c)


show(delta)
show(p12_numerator)
show(p12_denominator)
show(p12_numerator / p12_denominator)
plt.title("circle is $\\delta$, flat line is $d_{23}$ denominator, \nsloped line is $d_{23}$ numerator, $\infty$ is $d_{23}$ itself. \nx is set to its true value.")
plt.show()

show(delta)
show(p15_numerator)
show(p15_denominator)
show(p15_numerator / p15_denominator)
plt.title("circle is $\\delta$, gently ascending line is $d_{26}$ denominator, \nsteep line is $d_{26}$ numerator, $\infty$ is $d_{26}$ itself. \nx is set to its true value.")
plt.show()

A = 0.42593834 # new name W
show(delta)
show(A_numerator / -6)
show(A_denominator)
show(A_numerator / A_denominator / -6)
show(delta * A)
plt.title('''circle is $\\delta$, line is $W\\alpha$ denominator, red dot is $W\\alpha$ numerator, C-shape is $W\\alpha$ itself.
for reference, we also show the circle with radius $W$, that is where $W\\alpha$ is supposed to lie.
x is set to its true value.''')
plt.show()

exit()


show(delta)
show(p14_numerator)
show(p14_denominator)
show(p15_numerator)
show(p15_denominator)
delta = np.array([-0.12834049204788406+0.9917301639563593j])
show(delta)

plt.show()

'''
delta = np.linspace(0, 10, n)
p14 = -(2*sqrt(3)*delta + sqrt(3)*conjugate(delta) + 3*I*conjugate(delta) + 2 - 2*sqrt(3)*I)/(2*(sqrt(3)*delta/2 - 3*I*delta/2 + sqrt(3)*conjugate(delta) + 1 + sqrt(3)*I))
plt.plot(delta, np.abs(p14))
plt.show()
'''