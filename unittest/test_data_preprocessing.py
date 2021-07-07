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
        with open('dataset/airplane/airplane1.binvox', 'rb') as f:
            self.voxel_grid_airplane1 = binvox_rw.read_as_3d_array(f).data
        with open('dataset/airplane/airplane2.binvox', 'rb') as f:
            self.voxel_grid_airplane2 = binvox_rw.read_as_3d_array(f).data
        with open('dataset/lamp/lamp1.binvox', 'rb') as f:
            self.voxel_grid_lamp1 = binvox_rw.read_as_3d_array(f).data
        with open('dataset/lamp/lamp2.binvox', 'rb') as f:
            self.voxel_grid_lamp2 = binvox_rw.read_as_3d_array(f).data

    def test_get_reference_label(self, visualize=True):
        ref_label_chair1 = get_reference_label('dataset/chair/chair1.pts', 'dataset/chair/chair1.seg')
        ref_label_chair2 = get_reference_label('dataset/chair/chair2.pts', 'dataset/chair/chair2.seg')
        ref_label_airplane1 = get_reference_label('dataset/airplane/airplane1.pts', 'dataset/airplane/airplane1.seg')
        ref_label_airplane2 = get_reference_label('dataset/airplane/airplane2.pts', 'dataset/airplane/airplane2.seg')
        ref_label_lamp1 = get_reference_label('dataset/lamp/lamp1.pts', 'dataset/lamp/lamp1.seg')
        ref_label_lamp2 = get_reference_label('dataset/lamp/lamp2.pts', 'dataset/lamp/lamp2.seg')
        if visualize:
            visualization.visualize_label(ref_label_chair1)
            visualization.visualize_label(ref_label_chair2)
            visualization.visualize_label(ref_label_airplane1)
            visualization.visualize_label(ref_label_airplane2)
            visualization.visualize_label(ref_label_lamp1)
            visualization.visualize_label(ref_label_lamp2)
        return ref_label_chair1, ref_label_chair2, ref_label_airplane1, ref_label_airplane2, ref_label_lamp1, ref_label_lamp2

    def test_get_surface_label(self, visualize=True):
        res = self.test_get_reference_label(visualize=False)
        sur_label_chair1 = get_surface_label(self.voxel_grid_chair1, res[0])
        sur_label_chair2 = get_surface_label(self.voxel_grid_chair2, res[1])
        sur_label_airplane1 = get_surface_label(self.voxel_grid_airplane1, res[2])
        sur_label_airplane2 = get_surface_label(self.voxel_grid_airplane2, res[3])
        sur_label_lamp1 = get_surface_label(self.voxel_grid_lamp1, res[4])
        sur_label_lamp2 = get_surface_label(self.voxel_grid_lamp2, res[5])
        if visualize:
            visualization.visualize_label(sur_label_chair1)
            visualization.visualize_label(sur_label_chair2)
            visualization.visualize_label(sur_label_airplane1)
            visualization.visualize_label(sur_label_airplane2)
            visualization.visualize_label(sur_label_lamp1)
            visualization.visualize_label(sur_label_lamp2)
        return sur_label_chair1, sur_label_chair2, sur_label_airplane1, sur_label_airplane2, sur_label_lamp1, sur_label_lamp2

    def test_get_voxel_grid_label(self, visualize=True):
        k = 5
        res = self.test_get_surface_label(visualize=False)
        vox_grid_label_chair1 = get_voxel_grid_label(self.voxel_grid_chair1, res[0], k)
        vox_grid_label_chair2 = get_voxel_grid_label(self.voxel_grid_chair2, res[1], k)
        vox_grid_label_airplane1 = get_voxel_grid_label(self.voxel_grid_airplane1, res[2], k)
        vox_grid_label_airplane2 = get_voxel_grid_label(self.voxel_grid_airplane2, res[3], k)
        vox_grid_label_lamp1 = get_voxel_grid_label(self.voxel_grid_lamp1, res[4], k)
        vox_grid_label_lamp2 = get_voxel_grid_label(self.voxel_grid_lamp2, res[5], k)
        if visualize:
            visualization.visualize_label(vox_grid_label_chair1)
            visualization.visualize_label(vox_grid_label_chair2)
            visualization.visualize_label(vox_grid_label_airplane1)
            visualization.visualize_label(vox_grid_label_airplane2)
            visualization.visualize_label(vox_grid_label_lamp1)
            visualization.visualize_label(vox_grid_label_lamp2)
