# https://en.wikipedia.org/wiki/Mutually_unbiased_bases#The_problem_of_finding_a_maximal_set_of_MUBs_when_d_=_6
# code based on question
# https://stackoverflow.com/questions/64821151/notfounderror-when-using-a-tensorflow-optimizer-on-complex-variables-on-a-gpu

import sys
import tensorflow as tf
import numpy as np


run_id = int(sys.argv[1])
tf.random.set_seed(run_id)


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
    # This takes the Jaming et al parametrization to the Raynal et al one:
    ws = ws[:, [0, 3, 2, 5, 4, 1]] [[0, 3, 4, 1, 2, 5], :]
    return ws


def canonical_fourier_fn(x, y):
    WC = np.conjugate(W)
    f2 = tf.cast(tf.convert_to_tensor([[1, 1], [1, -1]]), dtype=np.complex128)
    tx = tf.convert_to_tensor([[1, x], [1, -x]], dtype=np.complex128)
    ty = tf.convert_to_tensor([[1, y], [1, -y]], dtype=np.complex128)
    l01 = tf.concat([f2, f2, f2], axis=1)
    l23 = tf.concat([tx, W * tx, WC * tx], axis=1)
    l45 = tf.concat([ty, WC * ty, W * ty], axis=1)
    l01 = tf.cast(l01, dtype=np.complex128)
    m = tf.concat([l01, l23, l45], axis=0)
    return m


def angler(x):
    return np.angle(x) * 180 / np.pi


def test_canonical_fourier():
    np.set_printoptions(precision=3, suppress=True, linewidth=100000)
    print("good")
    x = np.exp(7j/180*np.pi)
    y = np.exp(5j/180*np.pi)
    f = canonical_fourier(x, y)
    print(angler(f))
    xvar = tf.Variable(x, dtype=tf.complex128)
    yvar = tf.Variable(y, dtype=tf.complex128)
    f2 = canonical_fourier_fn(xvar, yvar)
    print(angler(f2))
    assert np.allclose(f, f2)


# test_canonical_fourier() ; exit()


# x = np.complex128(0.5718790679708813+0.8203379374481935j) # ours
# xs = np.complex128(np.exp(1.009j)) # raynal

optimal_logxs = tf.constant([124.88147886 / 360]) # taken from our run, 180-55.11852114
'''
x angles [124.88147886]
d angles [[180.73366437 308.0888661  104.20720639  68.0888661   60.73366437
  149.49680294]
 [ 48.02372117 324.14206146  48.02372117 160.66851944   9.43165801
  160.66851944]]

that's
d angles [[180.73366437 308.0888661  104.20720639  d[0, 1]-240   d[0, 0]-120
  149.49680294]
 [ 48.02372117 324.14206146  d[1, 0] 160.66851944   9.43165801 d[1, 3]]]

'''


SQ = np.sqrt(6)
identity = tf.eye(n, dtype=tf.complex128)


logds_init = tf.random.uniform([2, 6], dtype=tf.float64)
logds = tf.Variable(logds_init, trainable=True)


logxs_init = tf.random.uniform([6], dtype=tf.float64)
logxs = tf.Variable(logxs_init, trainable=True)

# the only free param:
logd1_p = tf.Variable(1.0, dtype=tf.float64, trainable=True)

logd2_init = tf.random.uniform([6], dtype=tf.float64)
logd2 = tf.Variable(logd2_init, trainable=True)



variables = [logxs, logd1_p, logd2]

def closeness(a, b):
    return tf.reduce_sum(tf.abs(a - b) ** 2)


def phase_fn():
    # 1/12 is 1j*conj(W). 1/2 is -1.
    logd1 = tf.convert_to_tensor(
        [logd1_p, -logd1_p, 2 * logd1_p + logxs[0] + 1/12,
        1/12 + 1/2 + logxs[0] - 2 * logd1_p,
        logd1_p, -logd1_p])
    logds = tf.stack([logd1, logd2])
    xs = tf.exp(tf.complex(0 * logxs, logxs) * 2 * np.pi)
    ds = tf.exp(tf.complex(0 * logds, logds) * 2 * np.pi)
    return xs, ds


def mub_fn():
    xs, ds = phase_fn()
    one_param = True
    if one_param:
        N1 = canonical_fourier_fn(1, xs[0])
        N2 = canonical_fourier_fn(xs[0], xs[0])
        N3 = canonical_fourier_fn(xs[0], 1)
    else:
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

opt = tf.keras.optimizers.SGD(learning_rate=0.001)

last_seen_loss = 100

for iteration in range(3000):
    with tf.GradientTape() as tape:
        mub = mub_fn()
        loss = loss_fn(mub)
    grads = tape.gradient(loss, variables)
    opt.apply_gradients(zip(grads, variables))
    if iteration % 100 == 0:
        print(iteration, loss.numpy())
        if loss.numpy() >= last_seen_loss - 0.01 and last_seen_loss > 0.1:
            print("terminating, not promising")
            exit()
        last_seen_loss = loss.numpy()


mub = mub_fn()
np.save("mub.npy", mub.numpy())


print("check out the current parametrization, these numbers might be ignored:")
print("x angles", logxs.numpy() * 360)
print("d angles", logds.numpy() * 360)

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
