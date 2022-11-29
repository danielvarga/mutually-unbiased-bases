# https://en.wikipedia.org/wiki/Mutually_unbiased_bases#The_problem_of_finding_a_maximal_set_of_MUBs_when_d_=_6
# code based on question
# https://stackoverflow.com/questions/64821151/notfounderror-when-using-a-tensorflow-optimizer-on-complex-variables-on-a-gpu

import tensorflow as tf
import numpy as np


from base import *


np.random.seed(1)

def random_phases(shape):
    return np.exp(2 * np.pi * 1j * np.random.uniform(size=shape))

x, y = random_phases((2, ))
F_np = canonical_fourier(x, y)
# F_np = tao() # this if you want Tao


assert np.allclose(np.abs(F_np), np.ones_like(F_np))
assert np.allclose(trans(F_np, F_np), 6 * np.eye(6))

F = tf.Variable(F_np, trainable=False)

n = 6
m = 1000

mat_r = tf.random.uniform([m, n], dtype=tf.float64)
mat_i = tf.random.uniform([m, n], dtype=tf.float64)
mat = 2 * tf.complex(mat_r, mat_i) - 1

var = tf.Variable(mat, trainable=True)

identity = tf.eye(n, dtype=tf.complex128)

def closeness(a, b):
    return tf.reduce_sum(tf.abs(a - b) ** 2)

def loss_fn():
    terms = []
    norm_squares = tf.reduce_sum(var * tf.math.conj(var), axis=1)
    terms.append(closeness(norm_squares, 6))

    terms.append(closeness(tf.abs(var), 1))

    prod = F @ tf.transpose(var, conjugate=True)
    terms.append(closeness(tf.abs(prod), 6 ** 0.5))
    return sum(terms)

opt = tf.keras.optimizers.SGD(learning_rate=0.01)

for iteration in range(15000):
    with tf.GradientTape() as tape:
        loss = loss_fn()
    grads = tape.gradient(loss, [var])
    opt.apply_gradients(zip(grads, [var]))
    if iteration % 100 == 0:
        print(iteration, loss.numpy())

mat = var.numpy()
for u in mat[:10]:
    print(np.abs(u), trans(u, u), np.abs(F_np @ np.conjugate(u)))


np.save("fortyeight.npy", mat)
# np.save("fortyeight_tao.npy", mat) # this if you want Tao
