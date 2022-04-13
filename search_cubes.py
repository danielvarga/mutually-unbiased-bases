# searching for Hadamard cubes.


# this variant keeps the matrices very close to the manifold of unitary matrices,
# thanks to the post-gradient update explained here:
# https://people.eecs.berkeley.edu/~wkahan/Math128/NearestQ.pdf

import sys
import tensorflow as tf
import numpy as np

run_id = int(sys.argv[1])

n = 6

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


mat_r = tf.random.uniform([n, n, n], dtype=tf.float64)
mat_i = tf.random.uniform([n, n, n], dtype=tf.float64)
mat = 2 * tf.complex(mat_r, mat_i) - 1
mat = mat / tf.complex(tf.abs(mat), tf.zeros((n, n, n), dtype=tf.float64)) / 6

var = tf.Variable(mat, trainable=True, constraint=conditional_many_procrustes)


def closeness(a, b):
    return tf.reduce_sum(tf.abs(a - b) ** 2)

target = tf.ones((n, n), dtype=tf.float64) / n ** 0.5

def hadamard_loss(u, alpha=1.0):
    how_constant = closeness(tf.abs(u), target)
    how_unitary = closeness(tf.transpose(u, conjugate=True) @ u, identity)
    return how_constant + alpha * how_unitary


def slice_2d(c, direction, coord):
    if direction == 0:
        return c[coord, :, :]
    elif direction == 1:
        return c[:, coord, :]
    elif direction == 2:
        return c[:, :, coord]
    assert False


def slice_1d(c, direction, coord1, coord2):
    if direction == 0:
        return c[coord1, coord2, :]
    elif direction == 1:
        return c[:, coord1, coord2]
    elif direction == 2:
        return c[coord2, :, coord1]
    assert False


def slice_1d_loss(v, alpha=1 / 6):
    return alpha * closeness(tf.reduce_sum(v), 1)


def loss_fn():
    alpha = 1.0
    terms = []
    for direction in range(3):
        for coord in range(n):
            s = slice_2d(var, direction, coord)
            l = hadamard_loss(s, alpha=alpha)
            terms.append(l)
    for direction in range(2):
        one_dim_sums = tf.reduce_sum(var, axis=direction)
        l = 1 / 36 * closeness(one_dim_sums, 1)
        terms.append(l)
    return sum(terms)

opt = tf.keras.optimizers.SGD(learning_rate=0.04)

iteration_count = 10001
iteration = 0
while iteration < iteration_count:
    with tf.GradientTape() as tape:
        loss = loss_fn()
    grads = tape.gradient(loss, [var])
    opt.apply_gradients(zip(grads, [var]))

    loss = loss.numpy()
    if iteration % 100 == 0:
        print(iteration, loss)
        sys.stdout.flush()
    if iteration == 5000:
        print("turning on projecting to unitaries")
        do_procrustes.assign(True)
    if iteration == 30000 and loss > 0.01:
        print("not promising, terminating")
        exit()
    iteration += 1


np.save("cube_%05d.two_1d_slices_summing_to_1.npy" % run_id, var.numpy())
