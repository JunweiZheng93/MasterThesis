from unittest import TestCase
from utils import visualization
import tensorflow as tf
import model
import numpy as np
import scipy.io


class TestModel(TestCase):

    def test_Resampling_get_voxel_value(self):

        # create input feature map in the shape of (B, num_parts, H, W, D, C)
        input_fmap = tf.range(96)
        input_fmap = tf.reshape(input_fmap, (2, 2, 2, 2, 2, 3))

        # create x, y, z coordinates in the shape of (B, num_parts, H, W, D)
        x = np.random.randint(0, 2, 2 * 2 * 2 * 2 * 2)
        x = tf.convert_to_tensor(x, dtype='int32')
        x = tf.reshape(x, [2, 2, 2, 2, 2])

        y = np.random.randint(0, 2, 2 * 2 * 2 * 2 * 2)
        y = tf.convert_to_tensor(y, dtype='int32')
        y = tf.reshape(y, [2, 2, 2, 2, 2])

        z = np.random.randint(0, 2, 2 * 2 * 2 * 2 * 2)
        z = tf.convert_to_tensor(z, dtype='int32')
        z = tf.reshape(z, [2, 2, 2, 2, 2])

        voxel_value = model.Resampling._get_voxel_value(input_fmap, x, y, z)

        self.assertEqual(input_fmap.shape, voxel_value.shape)

    def test_Resampling(self):

        my_model = model.Model(4)
        part1 = scipy.io.loadmat('datasets/chair_voxel/1c3f1a9cea91359c4c3e19c2c67c262f/part1.mat')['data'][:, :, :, np.newaxis]
        part2 = scipy.io.loadmat('datasets/chair_voxel/1c3f1a9cea91359c4c3e19c2c67c262f/part2.mat')['data'][:, :, :, np.newaxis]
        part3 = scipy.io.loadmat('datasets/chair_voxel/1c3f1a9cea91359c4c3e19c2c67c262f/part3.mat')['data'][:, :, :, np.newaxis]
        part4 = scipy.io.loadmat('datasets/chair_voxel/1c3f1a9cea91359c4c3e19c2c67c262f/part4.mat')['data'][:, :, :, np.newaxis]

        part1_trans = scipy.io.loadmat('datasets/chair_voxel/1c3f1a9cea91359c4c3e19c2c67c262f/part1_trans_matrix.mat')['data'][:3]
        part2_trans = scipy.io.loadmat('datasets/chair_voxel/1c3f1a9cea91359c4c3e19c2c67c262f/part2_trans_matrix.mat')['data'][:3]
        part3_trans = scipy.io.loadmat('datasets/chair_voxel/1c3f1a9cea91359c4c3e19c2c67c262f/part3_trans_matrix.mat')['data'][:3]
        part4_trans = scipy.io.loadmat('datasets/chair_voxel/1c3f1a9cea91359c4c3e19c2c67c262f/part4_trans_matrix.mat')['data'][:3]

        source = tf.cast(tf.expand_dims(tf.stack([part1, part2, part3, part4], axis=0), axis=0), dtype=tf.float32)
        gt_theta = tf.cast(
            tf.expand_dims(tf.stack([part1_trans, part2_trans, part3_trans, part4_trans], axis=0), axis=0),
            dtype=tf.float32)
        outputs = my_model.composer.stn.resampling([source, gt_theta]).numpy()
        # threshold can be changed in order to get better visualization result(similar to de-noising)
        outputs = np.where(outputs > 0.5, 1, 0)

        # show parts (you should check if the part is like the unscaled part)
        for part in outputs[0]:
            visualization.visualize(part[:, :, :, 0], show_grid=True, show_axis=True)

        # show shape (you should check if the part is like the unscaled shape)
        outputs = tf.squeeze(tf.reduce_sum(outputs, axis=1))
        outputs = tf.where(outputs >= 1, 1, 0)
        visualization.visualize(outputs, show_grid=True, show_axis=True)
