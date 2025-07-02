import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import MDS, TSNE

from base import *


def merge_canonized_cubes(folder_path, output_file):
    cube_list = []

    # Get list of .npy files and sort them for consistent order
    file_list = sorted(f for f in os.listdir(folder_path) if f.endswith('.npy'))
    
    for filename in file_list:
        filepath = os.path.join(folder_path, filename)
        data = np.load(filepath)
        if data.shape != (6, 6, 6):
            raise ValueError(f"File {filename} has shape {data.shape}, expected (6, 6, 6)")
        cube_list.append(data)
    
    merged_array = np.stack(cube_list, axis=0)  # Shape (N, 6, 6, 6)
    np.save(output_file, merged_array)
    print(f"Saved merged array of shape {merged_array.shape} to {output_file}")


# run it only once:
# merge_canonized_cubes(folder_path='canonized_cubes', output_file='merged_canonized_cubes.npy') ; exit()
# merge_canonized_cubes(folder_path='straight_cubes', output_file='merged_straight_cubes.npy') ; exit()

cubes = np.load("merged_canonized_cubes.npy") ; print(cubes.shape, "canonized cubes read")
# cubes = np.load("merged_straight_cubes.npy") ; print(cubes.shape, "straight cubes read")


szollosis = cubes[:, 0, :, :] # top Szollosi face of each cube
# merged_canonized_cubes.npy contains the deinterlaced variant a,b,c,d,e,f
# instead of Figure 1's interlaced a,f,b,e,c,d ordering.
A, B, C, D, E, F = szollosis[:, 0, :].T # each a 1d array indexed by the cubes.

# Legyen adva a Szöllősi mátrix olyan formában, ahogy a kanonizált alakú kockában van, azaz ahogy jelenleg a cikkünk 14. oldalán lévő ábra legfelső lapja mutatja.
# Legyen ezután x=a/b, y=b/c, és \alpha= x+y+(1/xy). Elvileg az \alpha paraméter lesz benne a "hatszög alakú" hipocikloid tartományban. 
x = A / B
y = B / C
alpha = x + y + 1 / x / y


cf = cubes.reshape((len(cubes), -1))
cf_real_part = cf.real
cf_imag_part = cf.imag

cube_embedding = PCA(n_components=3, random_state=47).fit_transform(cf_imag_part)
# -> turns out that this cleanly separates into 8 clusters.
cube_labels = KMeans(n_clusters=8, random_state=47).fit_predict(cube_embedding)

hypocycloid_embedding = PCA(n_components=2, random_state=42).fit_transform(cf_real_part)
# -> turns out that this quite cleanly cuts the hypocycloid into 6 segments.
hypocycloid_labels = KMeans(n_clusters=6, random_state=42).fit_predict(hypocycloid_embedding)

# this hopefully corresponds to a binary subtask, the 8-element discrete decomposing into [2]x[2]x[2].
# it worked for this random seed 47.
binary_labels = cube_embedding[:, :2].sum(axis=-1) > 0


def regression(X, Y, title):
    # Y = np.eye(8)[cube_labels]
    # Y = binary_labels
    # Y = np.stack((alpha.real, alpha.imag), axis=1)
    # Y = alpha.imag

    # X = cf_real_part

    from sklearn.linear_model import Ridge
    ridge = Ridge(alpha=0.01, fit_intercept=False)
    ridge.fit(X, Y)
    W = ridge.coef_.T
    pred = ridge.predict(X)
    pred2 = X @ W
    assert np.allclose(pred, pred2)

    rmse = np.mean((pred - Y) ** 2) ** 0.5
    print(f"ridge RMSE:", rmse)
    sparsity = 100 * np.mean(np.isclose(W, 0))
    print(f'ridge {sparsity:.1f}% of weights are zero')

    # we assume that the elements are of the form plusminus sqrt(k) / N, where N=24.
    N = 24
    W_round = np.around((W * N) ** 2)
    W_hat = np.sqrt(W_round) * np.sign(W) / N

    W_label = W_round * np.sign(W)
    W_label = W_label.reshape((6, 6, 6))

    print(W_label)

    print("max", np.abs(W - W_hat).max())

    pred = X @ W_hat

    rmse = np.mean((pred - Y) ** 2) ** 0.5
    print(f"ridge RMSE after rounding:", rmse)


    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plotting text labels at each (x, y, z) point
    offset = 0.1
    for x in range(W_label.shape[0]):
        for y in range(W_label.shape[1]):
            for z in range(W_label.shape[2]):
                value = int(W_label[x, y, z])
                ax.text(x - offset, y + offset, z + offset, str(value), fontsize=8, ha='right', va='top')
    x, y, z = np.indices(W_label.shape)
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    tab10 = plt.get_cmap('tab10')
    colors_discrete = tab10(W_label.flatten().astype(int) + 4)

    sc = ax.scatter(x, y, z, c=colors_discrete, s=40)
    ax.scatter(*np.indices(W_label.shape).reshape(3, -1), alpha=0.1, s=5)
    plt.title(title)
    ax.set_xlabel("Szollosi slices")
    ax.set_ylabel("Fourier 1 slices")
    ax.set_zlabel("Fourier 2 slices")
    ax.view_init(elev=10, azim=185)
    plt.tight_layout()
    plt.show()


regression(X=cf_real_part, Y=alpha.real, title="Weights for calculating Re(alpha) from the real part of the canonized cube")

regression(X=cf_real_part, Y=alpha.imag, title="Weights for calculating Im(alpha) from the real part of the canonized cube")


plt.scatter(hypocycloid_embedding[:, 0], hypocycloid_embedding[:, 1], s=2) #, c=hypocycloid_labels, cmap='tab10')
plt.gca().set_aspect('equal')
# plt.title("2D PCA and k-means of real part of cube")
plt.title("2D PCA of real part of cube")
plt.show()


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(cube_embedding[:, 0], cube_embedding[:, 1], cube_embedding[:, 2], c=cube_labels, cmap='tab10')
plt.title('3D PCA and k-means of imaginary part of cube\nIt is supposed to cluster into 8 categories perfectly.')
plt.show()


plt.scatter(alpha.real, alpha.imag, s=2)
plt.gca().set_aspect('equal')
plt.title("Szollosi's hypocycloid of alpha parameters")
plt.show()

plt.scatter(alpha.real, alpha.imag, s=6, c=cube_labels, cmap='tab10')
plt.gca().set_aspect('equal')
plt.title("Szollosi's hypocycloid of alpha parameters,\nlabeled by (imaginary part based) 8-clustering of the cubes.\nNo pattern.")
plt.show()

plt.scatter(alpha.real, alpha.imag, s=6, c=hypocycloid_labels, cmap='tab10')
plt.gca().set_aspect('equal')
plt.title("Szollosi's hypocycloid of alpha parameters,\nlabeled by the (real part based) 6-clustering PCA-based clustering of the cubes.\nClear pattern.")
plt.show()

plt.scatter(angler(A), angler(B), s=6, c=cube_labels, cmap='tab10')
plt.gca().set_aspect('equal')
plt.title("a and b, the Sz_11 and Sz_12 elements of the deinterlaced Szollosi matrix,\nlabeled by the 8-clustering of the cubes.\nThere is a pattern.")
plt.show()



exit()


N = 200
cubes = cubes[:N]
print(f"{N if N is not None else 'all'} kept")

cf = cubes.reshape((len(cubes), -1))
cf_real = np.concatenate([cf.real, cf.imag], axis=-1).reshape((len(cubes), -1))

'''
embedding = PCA(n_components=3).fit_transform(cf_real)
# -> turns out that this cleanly separates into 8 clusters.
labels = KMeans(n_clusters=8, random_state=42).fit_predict(embedding)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=labels, cmap='tab10')
plt.title('3D PCA')
plt.show()


print("filtering for cluster 0")
cubes = cubes[labels == 1]
cf = cubes.reshape((len(cubes), -1))
cf_real = np.concatenate([cf.real, cf.imag], axis=-1).reshape((len(cubes), -1))

embedding = PCA(n_components=2).fit_transform(cf_real)
plt.scatter(embedding[:, 0], embedding[:, 1])
plt.title('2D PCA of single cluster')
plt.show()


embedding = PCA(n_components=3).fit_transform(cf_real)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2])
plt.title('3D PCA of single cluster')
plt.show()

exit()
'''


dists_coordwise = np.abs(cf[:, None, :] - cf[None, :, :])
dists = np.linalg.norm(dists_coordwise, axis=-1)

print("calculating MDS")
embedding = MDS(n_components=2, dissimilarity='precomputed', random_state=42, normalized_stress='auto', metric=True).fit_transform(dists)
print("calculating MDS done")
plt.scatter(embedding[:, 0], embedding[:, 1])
plt.title('MDS Projection')
plt.show()


print("calculating TSNE")
embedding = TSNE(n_components=2, metric='precomputed', init='random', random_state=42).fit_transform(dists)
print("calculating TSNE done")
plt.scatter(embedding[:, 0], embedding[:, 1])
plt.title('t-SNE Projection')
plt.show()

closests = np.min(np.where(np.eye(dists.shape[0]), np.inf, dists), axis=-1)
plt.hist(closests, bins=100)
plt.show()
