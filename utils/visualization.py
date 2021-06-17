from matplotlib import pyplot as plt
import numpy as np


def visualization(voxel, label):

    # TODO: this function is not fully functional!!!

    color_bar = ['white', 'red', 'green', 'blue', 'pink', 'yellow', 'purple']
    color = np.empty(label.shape, dtype=str)
    num_parts = np.max(label)
    for part in range(1, num_parts + 1):
        color[label == part] = color_bar[part]

    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(voxel, edgecolor='k', facecolors=color)
    plt.axis('off')
    plt.grid('off')
    plt.show()
