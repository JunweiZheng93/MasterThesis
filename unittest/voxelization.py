from unittest import TestCase
from utils.voxelization import *
from utils.visualization import *


class TestVoxelization(TestCase):

    def test_voxelization(self):
        point_cloud = np.genfromtxt('dataset/1.pts')
        points_label = np.genfromtxt('dataset/1.seg')
        final_voxel, final_label = voxelization(point_cloud, points_label)
        visualization(final_voxel, final_label)
