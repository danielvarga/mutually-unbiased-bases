# https://en.wikipedia.org/wiki/Mutually_unbiased_bases#The_problem_of_finding_a_maximal_set_of_MUBs_when_d_=_6
# code based on question
# https://stackoverflow.com/questions/64821151/notfounderror-when-using-a-tensorflow-optimizer-on-complex-variables-on-a-gpu

import tensorflow as tf
import numpy as np


n = 6
m = 4

mat_r = tf.random.uniform([m, n, n], dtype=tf.float64)
mat_i = tf.random.uniform([m, n, n], dtype=tf.float64)
mat = 2 * tf.complex(mat_r, mat_i) - 1

var = tf.Variable(mat, trainable=True)

identity = tf.eye(n, dtype=tf.complex128)

def closeness(a, b):
    return tf.reduce_sum(tf.abs(a - b) ** 2)

def loss_fn():
    terms = []
    for i in range(m):
        u = var[i]
        terms.append(closeness(tf.transpose(u, conjugate=True) @ u, identity))

    target = tf.ones((n, n), dtype=tf.float64) / n ** 0.5
    for i in range(m):
        for j in range(i + 1, m):
            prod = tf.transpose(var[i], conjugate=True) @ var[j]
            terms.append(closeness(tf.abs(prod), target))
    return sum(terms)

opt = tf.keras.optimizers.SGD(learning_rate=0.01)

for iteration in range(1000):
    with tf.GradientTape() as tape:
        loss = loss_fn()
    grads = tape.gradient(loss, [var])
    opt.apply_gradients(zip(grads, [var]))
    if iteration % 100 == 0:
        print(iteration, loss.numpy())


np.save("mub.npy", var.numpy())

u0 = var[0].numpy()
u1 = var[1].numpy()

print("----\n|A† A|")
print(np.abs(np.conjugate(u0.T) @ u0))
print("----\n√n |A† B|")
print(n ** 0.5 * np.abs(np.conjugate(u0.T) @ u1))
print("----\nA")
print(u0)
print("----\nB")
print(u1)
