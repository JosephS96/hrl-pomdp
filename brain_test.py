import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def f(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))


if __name__ == "__main__":
    print("brian test")

    spatial = '/Users/josesanchez/Documents/IAS/Master-Thesis/Brain_Networks/spatial.npy'
    file = np.load(spatial)

    example = file[10]
    example = example.astype(np.float32)

    x = []
    y = []
    z = []

    for data in example:
        x.append(data[0])
        y.append(data[1])
        z.append(data[2])

    ax = plt.axes(projection='3d')

    ax.scatter3D(xs=x, ys=y, zs=z, cmap='viridis')
    # ax.plot_surface(X=X, Y=Y, Z=Z, cmap='viridis', edgecolor='none')
    # ax.plot_trisurf(X=x, Y=y, Z=z, cmap='viridis', edgecolor='none')



    plt.show()

