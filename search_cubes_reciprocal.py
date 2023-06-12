# searching for Hadamard cubes.


# this variant keeps the matrices very close to the manifold of unitary matrices,
# thanks to the post-gradient update explained here:
# https://people.eecs.berkeley.edu/~wkahan/Math128/NearestQ.pdf

import sys
import tensorflow as tf
import numpy as np
from itertools import combinations


n, run_id = map(int, sys.argv[1:])


tf.random.set_seed(run_id)

identity = tf.eye(n, dtype=tf.complex128)


mat_r = tf.random.uniform([n, n, n], dtype=tf.float64)
mat_i = tf.random.uniform([n, n, n], dtype=tf.float64)
mat = 2 * tf.complex(mat_r, mat_i) - 1
mat = mat / tf.complex(tf.abs(mat), tf.zeros((n, n, n), dtype=tf.float64)) / n ** 0.5

var = tf.Variable(mat, trainable=True)


c = np.load("canonized_cubes/canonized_cube_00000.npy")
c = np.random.permutation(c.flatten()).reshape((6, 6, 6))
# c += np.random.normal(size=c.shape, scale=0.1)
var = tf.Variable(c, trainable=True)




checkered_np = np.ones((2, 2, 2), dtype=np.complex128)
for i in range(2):
    for j in range(2):
        for k in range(2):
            checkered_np[i, j, k] = 2 * ((i+j+k) % 2) - 1
checkered = tf.constant(checkered_np)


def closeness(a, b):
    return tf.reduce_sum(tf.abs(a - b) ** 2)

target = tf.ones((n, n), dtype=tf.float64) / n ** 0.5


def haagerup_loss_on_minicube(mc):
    left  = mc[0, 0, 0] * mc[0, 1, 1] * mc[1, 1, 0] * mc[1, 0, 1]
    right = mc[1, 1, 1] * mc[1, 0, 0] * mc[0, 0, 1] * mc[0, 1, 0]
    return closeness(left, right)


def fast_haagerup_loss_on_minicube(mc):
    prod = tf.math.reduce_prod(mc ** checkered)
    return closeness(prod, 1)


# mcd as in minicube descriptor, a triplet of coordinate pairs.
def select_minicube(c, mcd):
    mc = np.empty((2, 2, 2), dtype=object)
    for i, x in enumerate(mcd[0]):
        for j, y in enumerate(mcd[1]):
            for k, z in enumerate(mcd[2]):
                mc[i, j, k] = c[x, y, z]
    return mc


def all_haagerup_losses(c):
    losses = []
    pairs = list(combinations(range(6), 2))
    for mcd0 in pairs:
        for mcd1 in pairs:
            for mcd2 in pairs:
                mc = select_minicube(c, [mcd0, mcd1, mcd2])
                l = haagerup_loss_on_minicube(mc)
                losses.append(l)
    return losses


def contiguous_haagerup_losses(c):
    losses = []
    for i in range(n - 1):
        for j in range(n - 1):
            for k in range(n - 1):
                mc = c[i: i+2, j: j+2, k: k+2]
                l = fast_haagerup_loss_on_minicube(mc)
                losses.append(l)
    return losses


def reciprocal_transpose(u):
    ut = tf.transpose(u, conjugate=False)
    return tf.math.reciprocal(ut)


def matolcsi_loss(u):
    how_unitary = closeness(reciprocal_transpose(u) @ u, identity)
    return how_unitary


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
            l = matolcsi_loss(s)
            terms.append(l)
    for direction in range(3):
        one_dim_sums = tf.reduce_sum(var, axis=direction)
        l = 1 / 36 * closeness(one_dim_sums, 1)
        terms.append(l)

    # terms += all_haagerup_losses(var)
    terms += contiguous_haagerup_losses(var)
    return sum(terms)

opt = tf.keras.optimizers.SGD(learning_rate=0.001)

iteration_count = 205
iteration = 0
while iteration < iteration_count:
    with tf.GradientTape() as tape:
        loss = loss_fn()
    grads = tape.gradient(loss, [var])
    opt.apply_gradients(zip(grads, [var]))

    loss = loss.numpy()
    if (iteration % 10 == 0) or (iteration>=198) :
        print(iteration, loss)

        v = var.numpy()
        s = v[0, :, :]
        adj = np.conjugate(s.T) @ s

        # print(iteration, "abs loss", np.abs(v)[0, 0, :])
        # print(iteration, "adj loss", adj[:, :])

        sys.stdout.flush()

    iteration += 1


np.save("cube_d%d.%05d.reciprocal.npy" % (n, run_id), var.numpy())
