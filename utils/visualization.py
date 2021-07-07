from matplotlib import pyplot as plt
import numpy as np


def visualize_label(label, show_axis=False, show_grid=False):

    color_bar = ['white', 'red', 'green', 'blue', 'yellow']
    color = np.empty(label.shape, dtype=str)
    num_parts = np.max(label)
    for part in range(1, num_parts + 1):
        color[label == part] = color_bar[part]

    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(label > 0, edgecolor='k', facecolors=color)
    if not show_axis:
        plt.axis('off')
    if not show_grid:
        plt.grid('off')
    plt.show()


def visualize_voxel_grid(voxel_grid, show_axis=False, show_grid=False):
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(voxel_grid, edgecolor='k')
    if not show_axis:
        plt.axis('off')
    if not show_grid:
        plt.grid('off')
    plt.show()
