from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import numpy as np


def visualize(x,
              max_num_parts=8,
              show_fig=True,
              show_axis=False,
              show_grid=False,
              cmap='Set2',
              save_fig=False,
              save_dir=None):
    """
    :param x: input data to be visualized.
    :param max_num_parts: maximal number of parts of the category. e.g. the maximal number of parts where the
           category chair can be divided is 4.
    :param show_fig: show the figure or not.
    :param show_axis: show the axis of the figure or not.
    :param show_grid: show the grid of the figure or not.
    :param cmap: name of the cmap.
    :param save_fig: save the figure or not.
    :param save_dir: directory where the image is to be saved.
    """

    if show_fig:
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(projection='3d')
        new_cmap = get_cmap(max_num_parts, cmap)
        label_color = np.take(new_cmap, x, axis=0)
        ax1.voxels(x, facecolors=label_color)
        if not show_axis:
            plt.axis('off')
        if not show_grid:
            plt.grid('off')
        plt.show()
        plt.close(fig1)

    if save_fig:
        x = np.transpose(x, (0, 2, 1))
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(projection='3d')
        new_cmap = get_cmap(max_num_parts, cmap)
        label_color = np.take(new_cmap, x, axis=0)
        ax2.voxels(x, facecolors=label_color)
        if not show_axis:
            plt.axis('off')
        if not show_grid:
            plt.grid('off')
        plt.savefig(save_dir)
        plt.close(fig2)


def get_cmap(num_points, cmap):
    selected_cmap = cm.get_cmap(cmap, num_points)
    if not isinstance(selected_cmap, ListedColormap):
        raise ValueError(f'cmap should be <class \'matplotlib.colors.ListedColormap\'>, but got {type(selected_cmap)}')
    new_cmap = np.ones((num_points + 1, 4))
    new_cmap[1:] = selected_cmap.colors
    return new_cmap
