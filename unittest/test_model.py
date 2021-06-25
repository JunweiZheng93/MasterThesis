from unittest import TestCase
from model import *
import tensorflow as tf
import numpy as np


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

        voxel_value = Resampling._get_voxel_value(input_fmap, x, y, z)

        self.assertEqual(input_fmap.shape, voxel_value.shape)
