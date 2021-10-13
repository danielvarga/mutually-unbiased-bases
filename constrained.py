# https://en.wikipedia.org/wiki/Mutually_unbiased_bases#The_problem_of_finding_a_maximal_set_of_MUBs_when_d_=_6
# code based on question
# https://stackoverflow.com/questions/64821151/notfounderror-when-using-a-tensorflow-optimizer-on-complex-variables-on-a-gpu

# this variant keeps the matrices very close to the manifold of unitary matrices,
# thanks to the post-gradient update explained here:
# https://people.eecs.berkeley.edu/~wkahan/Math128/NearestQ.pdf

import sys
import tensorflow as tf
import numpy as np

run_id = int(sys.argv[1])

n = 6
m = 4

tf.random.set_seed(run_id)

identity = tf.eye(n, dtype=tf.complex128)

do_procrustes = tf.Variable(False, trainable=False)


# this only works well if x is already close to being unitary,
# so we only turn this on after the soft constraints bring it there.
def procrustes(x):
    y = tf.transpose(x, conjugate=True) @ x - identity
    return x - 1/2 * x @ y

def many_procrustes(xs):
    return tf.stack([procrustes(xs[i]) for i in range(xs.shape[0])])

def conditional_many_procrustes(xs):
    return tf.cond(do_procrustes, lambda: many_procrustes(xs), lambda: xs)


mat_r = tf.random.uniform([m, n, n], dtype=tf.float64)
mat_i = tf.random.uniform([m, n, n], dtype=tf.float64)
mat = 2 * tf.complex(mat_r, mat_i) - 1

var = tf.Variable(mat, trainable=True, constraint=conditional_many_procrustes)


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

opt = tf.keras.optimizers.SGD(learning_rate=0.002)

for iteration in range(20001):
    with tf.GradientTape() as tape:
        loss = loss_fn()
    grads = tape.gradient(loss, [var])
    opt.apply_gradients(zip(grads, [var]))

    if iteration % 1000 == 0:
        print(iteration, loss.numpy())
        sys.stdout.flush()
    if iteration == 1000:
        print("turning on projecting to unitaries")
        do_procrustes.assign(True)


np.save("mub_%03d.npy" % run_id, var.numpy())

'''
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
'''
