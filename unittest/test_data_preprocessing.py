from utils.data_preprocessing import *
from utils import visualization
from utils import binvox_rw
from unittest import TestCase


class TestDataPreprocessing(TestCase):

    def setUp(self) -> None:
        with open('dataset/chair/chair1.binvox', 'rb') as f:
            self.voxel_grid_chair1 = binvox_rw.read_as_3d_array(f).data
        with open('dataset/chair/chair2.binvox', 'rb') as f:
            self.voxel_grid_chair2 = binvox_rw.read_as_3d_array(f).data

    def test_get_reference_label(self, visualize=True):
        ref_label_chair1 = get_reference_label('dataset/chair/chair1.pts', 'dataset/chair/chair1.seg')
        ref_label_chair2 = get_reference_label('dataset/chair/chair2.pts', 'dataset/chair/chair2.seg')
        if visualize:
            visualization.visualize(ref_label_chair1)
            visualization.visualize(ref_label_chair2)
        return ref_label_chair1, ref_label_chair2

    def test_get_surface_label(self, visualize=True):
        res = self.test_get_reference_label(visualize=False)
        sur_label_chair1 = get_surface_label(self.voxel_grid_chair1, res[0])
        sur_label_chair2 = get_surface_label(self.voxel_grid_chair2, res[1])
        if visualize:
            visualization.visualize(sur_label_chair1)
            visualization.visualize(sur_label_chair2)
        return sur_label_chair1, sur_label_chair2

    def test_get_voxel_grid_label(self, visualize=True):
        res = self.test_get_surface_label(visualize=False)
        vox_grid_label_chair1 = get_voxel_grid_label(self.voxel_grid_chair1, res[0])
        vox_grid_label_chair2 = get_voxel_grid_label(self.voxel_grid_chair2, res[1])
        if visualize:
            visualization.visualize(vox_grid_label_chair1)
            visualization.visualize(vox_grid_label_chair2)
        return vox_grid_label_chair1, vox_grid_label_chair2

    def test_get_seperated_part_and_transformation(self, visualize=True):
        res = self.test_get_voxel_grid_label(visualize=False)
        part_voxel_grid_chair1, transformation_matrix_chair1 = get_seperated_part_and_transformation(res[0])
        part_voxel_grid_chair2, transformation_matrix_chair2 = get_seperated_part_and_transformation(res[1])
        if visualize:
            for part, transformation in zip(part_voxel_grid_chair1, transformation_matrix_chair1):
                visualization.visualize(part, show_axis=True, show_grid=True)
                print(transformation)
            for part, transformation in zip(part_voxel_grid_chair2, transformation_matrix_chair2):
                visualization.visualize(part, show_grid=True, show_axis=True)
                print(transformation)
