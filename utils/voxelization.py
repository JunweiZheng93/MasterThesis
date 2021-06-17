import numpy as np


def voxelization(point_cloud, points_label, resolution=32):

    # TODO: this function is not fully functional!!!

    """
    :param point_cloud: ndarray. the shape of the point cloud should be n*3, where n is the number of the points.
    :param points_label: ndarray. the semantic label of every points.
    :param resolution: int. the resolution of the final voxel. e.g. 32 means 32*32*32
    :return: tuple of ndarray. the output voxel and the corresponding labels.
    """
    h_idxes = []
    final_voxel = np.full(resolution**3, False)
    final_label = np.zeros(resolution**3, dtype=int)

    maxi = np.max(point_cloud, axis=0)
    mini = np.min(point_cloud, axis=0)
    voxel_size = np.max(np.ceil((maxi - mini) * 100 / resolution) / 100)
    for point in point_cloud:  # calculate which point in which voxel grid
        xyz_idx = (point - mini) // voxel_size
        h_idx = xyz_idx[0] + xyz_idx[1] * resolution + xyz_idx[2] * resolution ** 2
        h_idxes.append(h_idx)

    point_cloud_idxes = np.argsort(np.asarray(h_idxes))
    voxel_has_points, counts = np.unique(h_idxes, return_counts=True)
    voxel_has_points = voxel_has_points.astype(int)

    final_voxel[voxel_has_points] = True
    final_voxel = final_voxel.reshape([resolution, resolution, resolution])

    idx_start = 0
    for voxel_idex, count in zip(voxel_has_points, counts):
        point_cloud_idxes_per_voxel = point_cloud_idxes[idx_start:idx_start+count]
        idx_start += count
        votes = points_label[point_cloud_idxes_per_voxel]
        labels, vote_counts = np.unique(votes, return_counts=True)
        max_vote = np.argsort(vote_counts)[-1]
        voxel_label = labels[max_vote]
        final_label[voxel_idex] = voxel_label
    final_label = final_label.reshape([resolution, resolution, resolution])

    return final_voxel, final_label
