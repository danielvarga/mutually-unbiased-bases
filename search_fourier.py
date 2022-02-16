# https://en.wikipedia.org/wiki/Mutually_unbiased_bases#The_problem_of_finding_a_maximal_set_of_MUBs_when_d_=_6
# code based on question
# https://stackoverflow.com/questions/64821151/notfounderror-when-using-a-tensorflow-optimizer-on-complex-variables-on-a-gpu

import tensorflow as tf
import numpy as np


n = 6
m = 4


TP = 2 * np.pi
TIP = 2j * np.pi
W = np.exp(TIP / 3)


def xy_test():
    ws = np.ones((6, 6), dtype=np.complex128)
    ws[1:6:3, 1:6:2] *= 2
    ws[2:6:3, 1:6:2] *= 3
    print(ws.real)
    x = tf.constant(2, dtype=np.complex128)
    y = tf.constant(3, dtype=np.complex128)
    xxx = tf.stack([1, x, 1, x, 1, x], axis=0)
    yyy = tf.stack([1, y, 1, y, 1, y], axis=0)
    ooo = tf.ones([6], dtype=np.complex128)
    m = tf.stack([ooo, xxx, yyy, ooo, xxx, yyy], axis=0)
    print(m.numpy().real)


# xy_test() ; exit()


def canonical_fourier(x, y):
    ws = np.ones((6, 6), dtype=np.complex128)
    ws[1:6:3, 1:6:2] *= x
    ws[2:6:3, 1:6:2] *= y
    sixth = - np.conjugate(W)
    for i in range(1, 6):
        for j in range(1, 6):
            ws[i, j] *= sixth ** ((i * j) % 6)
    return ws


fourier_base = tf.constant(canonical_fourier(1, 1))

def canonical_fourier_fn(x, y):
    xxx = tf.cast(tf.stack([1, x, 1, x, 1, x], axis=0), dtype=np.complex128)
    yyy = tf.cast(tf.stack([1, y, 1, y, 1, y], axis=0), dtype=np.complex128)
    ooo = tf.ones([6], dtype=np.complex128)
    m = tf.stack([ooo, xxx, yyy, ooo, xxx, yyy], axis=0)
    return m * fourier_base # elementwise_mul


# x = np.complex128(0.5718790679708813+0.8203379374481935j)
xs = np.complex128(np.exp(1.009j))

SQ = np.sqrt(6)
identity = tf.eye(n, dtype=tf.complex128)


dmat_r = tf.random.uniform([2, 6], dtype=tf.float64)
dmat_i = tf.random.uniform([2, 6], dtype=tf.float64)
dmat = 2 * tf.complex(dmat_r, dmat_i) - 1
ds = tf.Variable(dmat, trainable=True)

xmat_r = tf.random.uniform([6], dtype=tf.float64)
xmat_i = tf.random.uniform([6], dtype=tf.float64)
xmat = 2 * tf.complex(xmat_r, xmat_i) - 1
xs = tf.Variable(xmat, trainable=True)


def closeness(a, b):
    return tf.reduce_sum(tf.abs(a - b) ** 2)


def mub_fn(xs, ds):
    N1 = canonical_fourier_fn(xs[0], xs[1])
    N2 = canonical_fourier_fn(xs[2], xs[3])
    N3 = canonical_fourier_fn(xs[4], xs[5])
    D_a = tf.linalg.diag(ds[0])
    D_b = tf.linalg.diag(ds[1])
    M1 = D_a @ N1 / SQ
    M2 = N2 / SQ
    M3 = D_b @ N3 / SQ
    mub = tf.stack([identity, M1, M2, M3])
    return mub


def loss_fn(mub):
    terms = []
    for i in range(m):
        u = mub[i]
        terms.append(closeness(tf.transpose(u, conjugate=True) @ u, identity))

    target = tf.ones((n, n), dtype=tf.float64) / n ** 0.5
    for i in range(m):
        for j in range(i + 1, m):
            prod = tf.transpose(mub[i], conjugate=True) @ mub[j]
            terms.append(closeness(tf.abs(prod), target))
    return sum(terms)

opt = tf.keras.optimizers.SGD(learning_rate=0.05)

for iteration in range(3000):
    with tf.GradientTape() as tape:
        mub = mub_fn(xs, ds)
        loss = loss_fn(mub)
    grads = tape.gradient(loss, [xs, ds])
    opt.apply_gradients(zip(grads, [xs, ds]))
    if iteration % 100 == 0:
        print(iteration, loss.numpy())


mub = mub_fn(xs, ds)
np.save("mub.npy", mub.numpy())

print("abs", np.abs(xs.numpy()), np.abs(ds.numpy()))


def angler(x):
    return np.angle(x) * 180 / np.pi


print("x angles", angler(xs.numpy()))
print("d angles", angler(ds.numpy()))

u0 = mub[1].numpy()
u1 = mub[2].numpy()

print("----\n|A† A|")
print(np.abs(np.conjugate(u0.T) @ u0))
print("----\n√n |A† B|")
print(np.abs(np.conjugate(u0.T) @ u1))
print("----\nA")
print(u0)
print("----\nB")
print(u1)
