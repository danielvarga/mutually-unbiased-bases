import sys
import numpy as np
import matplotlib.pyplot as plt

# python analyze_fouriers.py < fourier_bases

# typical line
# filename normalized/mub_100.npy basis 1 fourier_params_in_degrees 0.0005514574849350037 55.1187968716524 haagerup_distance 4.014721294667324e-13

mubs = []
mub = []
filenames = []
for l in sys.stdin:
    a = l.strip().split()
    filename = a[1]
    basis = int(a[3])
    # in degrees between 0 and 180.
    d1, d2 = map(float, a[5:7])
    # asserts a 1 2 3 1 2 3 ordering in the input
    mub.append([d1, d2])
    if basis == 3:
        mubs.append(mub)
        mub = []
        filenames.append(filename)

mubs = np.array(mubs)
mubs = mubs.reshape((len(mubs), -1))

fracs, _ = np.modf(mubs / 5 + 0.5)
fracs -= 0.5
fracs *= 5 # modulo 5
plt.hist(fracs.flatten(), bins=50)
plt.show()



mubs = np.exp(1j * mubs / 180 * np.pi)
mubs = mubs.flatten()
plt.scatter(np.real(mubs), np.imag(mubs))
plt.show()
