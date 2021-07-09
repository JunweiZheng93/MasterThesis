import numpy as np
import os


def get_reference_label(pcd_fp, label_fp, resolution=32):
    """
    :param pcd_fp: file path of the point cloud data
    :param label_fp: file path of the corresponding label to the point cloud data
    :param resolution: resolution of the voxel grid that you want to generate
    :return: voxel grid label generated from point cloud label
    """
    pcd = np.genfromtxt(pcd_fp, dtype='float32')
    pcd_label = np.genfromtxt(label_fp, dtype='int32')
    maxi = np.max(pcd, axis=0)
    mini = np.min(pcd, axis=0)
    voxel_size = np.max(np.ceil((maxi - mini) * 10000 / (resolution - 1)) / 10000)
    translation = np.max(pcd) + 0.5 * voxel_size
    pcd += translation

    voxel_grid_label_dict = dict()
    for point, point_label in zip(pcd, pcd_label):
        idx = point // voxel_size
        voxel_idx = int(idx[0] + idx[1] * resolution + idx[2] * resolution ** 2)
        if voxel_idx not in list(voxel_grid_label_dict.keys()):
            voxel_grid_label_dict[voxel_idx] = [point_label]
        else:
            voxel_grid_label_dict[voxel_idx].append(point_label)

    voxel_grid_label = np.full((resolution, resolution, resolution), 0, dtype='int32')
    for voxel_idx in list(voxel_grid_label_dict.keys()):
        count_list = voxel_grid_label_dict[voxel_idx]
        voxel_label = max(set(count_list), key=count_list.count)
        idx = [voxel_idx % resolution, voxel_idx // resolution % resolution, voxel_idx // resolution // resolution % resolution]
        voxel_grid_label[idx[0], idx[1], idx[2]] = voxel_label
    return voxel_grid_label


def get_surface_label(voxel_grid, reference_label):
    """
    :param voxel_grid: voxel grid generated from binvox file
    :param reference_label: voxel grid label generated from point cloud label
    :return: surface label for the voxel grid generated from binvox file
    """
    voxel_grid_cord = np.stack(np.where(voxel_grid), axis=1)
    ref_label_cord = np.stack(np.where(reference_label > 0), axis=1)
    surface_label = np.zeros_like(voxel_grid, dtype='int32')
    for cord in ref_label_cord:
        dist = cord - voxel_grid_cord
        mini_dist_idx = np.argmin(np.linalg.norm(dist, axis=1))
        surface_label_idx = voxel_grid_cord[mini_dist_idx]
        surface_label[surface_label_idx[0], surface_label_idx[1], surface_label_idx[2]] = reference_label[cord[0], cord[1], cord[2]]
    return surface_label


def get_voxel_grid_label(voxel_grid, surface_label, k=3):
    """
    :param voxel_grid: voxel grid generated from binvox file
    :param surface_label: surface label for the voxel grid generated from binvox file
    :param k: knn parameter for calculating the inner voxel label
    :return: voxel grid label (surface and inner voxels are all labeled)
    """
    voxel_grid_cord = np.stack(np.where(voxel_grid), axis=1)
    surface_label_cord = np.stack(np.where(surface_label > 0), axis=1)
    for cord in voxel_grid_cord:
        if list(cord) not in surface_label_cord.tolist():
            candidate_label_list = []
            dist = cord - surface_label_cord
            mini_dist_idx = np.argpartition(np.linalg.norm(dist, axis=1), k)[:k]
            candidate_cord = surface_label_cord[mini_dist_idx]
            for each in candidate_cord:
                candidate_label_list.append(surface_label[each[0], each[1], each[2]])
            voxel_label = max(set(candidate_label_list), key=candidate_label_list.count)
            surface_label[cord[0], cord[1], cord[2]] = voxel_label
            surface_label_cord = np.stack(np.where(surface_label > 0), axis=1)
    return surface_label


def check_file_path(pcd_fp, binvox_fp):

    # check pcd has corresponding label and img
    pcd_dir = os.path.join(pcd_fp, 'points')
    pcd_label_dir = os.path.join(pcd_fp, 'points_label')
    pcd_img_dir = os.path.join(pcd_fp, 'seg_img')
    pcd_names = [os.path.splitext(path)[0] for path in os.listdir(pcd_dir)]
    pcd_names.sort()
    label_names = [os.path.splitext(path)[0] for path in os.listdir(pcd_label_dir)]
    label_names.sort()
    img_names = [os.path.splitext(path)[0] for path in os.listdir(pcd_img_dir)]
    img_names.sort()
    assert pcd_names == label_names == img_names

    # check every pcd has its corresponding binvox
    binvox_names = os.listdir(binvox_fp)
    for pcd_name in pcd_names:
        assert pcd_name in binvox_names
