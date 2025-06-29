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


cf = cubes.reshape((len(cubes), -1))
cf_real_part = cf.real
cf_imag_part = cf.imag

cube_embedding = PCA(n_components=3).fit_transform(cf_imag_part)
# -> turns out that this cleanly separates into 8 clusters.
cube_labels = KMeans(n_clusters=8, random_state=42).fit_predict(cube_embedding)

hypocycloid_embedding = PCA(n_components=2).fit_transform(cf_real_part)
# -> turns out that this quite cleanly cuts the hypocycloid into 6 segments.
hypocycloid_labels = KMeans(n_clusters=6, random_state=42).fit_predict(hypocycloid_embedding)

plt.scatter(hypocycloid_embedding[:, 0], hypocycloid_embedding[:, 1], s=2, c=hypocycloid_labels, cmap='tab10')
plt.gca().set_aspect('equal')
plt.title("2D PCA and k-means of real part of cube")
plt.show()


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(cube_embedding[:, 0], cube_embedding[:, 1], cube_embedding[:, 2], c=cube_labels, cmap='tab10')
plt.title('3D PCA and k-means of imaginary part of cube\nIt is supposed to cluster into 8 categories perfectly.')
plt.show()


szollosis = cubes[:, 0, :, :] # top Szollosi face of each cube
# merged_canonized_cubes.npy contains the deinterlaced variant a,b,c,d,e,f
# instead of Figure 1's interlaced a,f,b,e,c,d ordering.
A, B, C, D, E, F = szollosis[:, 0, :].T # each a 1d array indexed by the cubes.

# Legyen adva a Szöllősi mátrix olyan formában, ahogy a kanonizált alakú kockában van, azaz ahogy jelenleg a cikkünk 14. oldalán lévő ábra legfelső lapja mutatja.
# Legyen ezután x=a/b, y=b/c, és \alpha= x+y+(1/xy). Elvileg az \alpha paraméter lesz benne a "hatszög alakú" hipocikloid tartományban. 
x = A / B
y = B / C
alpha = x + y + 1 / x / y

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
