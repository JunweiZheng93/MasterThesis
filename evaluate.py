import tensorflow as tf
import model
import scipy.io
import numpy as np
import os
import argparse
from utils import visualization


PROJ_ROOT = os.path.abspath(__file__)[:-11]
CATEGORY_MAP = {'chair': '03001627', 'table': '04379243', 'airplane': '02691156', 'lamp': '03636649'}


def evaluate_model(model_path,
                   mode='batch',
                   category='chair',
                   num_to_visualize=4,
                   single_shape_path=None,
                   H=32,
                   W=32,
                   D=32,
                   C=1):

    if mode == 'batch':
        num_parts = 3 if category == 'table' else 4
    elif mode == 'single':
        if not single_shape_path.endswith('/'):
            single_shape_path += '/'
        category_code = single_shape_path.split('/')[-3]
        category = list(CATEGORY_MAP.keys())[list(CATEGORY_MAP.values()).index(category_code)]
        num_parts = 3 if category == 'table' else 4
    else:
        raise ValueError('mode should be one of batch or single!')

    warm_up_data = tf.ones((1, H, W, D, C), dtype=tf.float32)
    my_model = model.Model(num_parts=num_parts)
    my_model.choose_training_process(1)
    my_model(warm_up_data)
    my_model.load_weights(model_path)

    dataset_path = os.path.join(PROJ_ROOT, 'datasets', CATEGORY_MAP[category])

    if mode == 'batch':
        all_shapes = os.listdir(dataset_path)
        idx = np.random.choice(len(all_shapes), num_to_visualize, replace=False)
        shapes_to_visualize = [all_shapes[each] for each in idx]

        for shape in shapes_to_visualize:

            gt_label_path = os.path.join(dataset_path, shape, 'object_labeled.mat')
            gt_label = scipy.io.loadmat(gt_label_path)['data']
            visualization.visualize(gt_label)

            shape_path = os.path.dirname(gt_label_path)
            gt_shape_path = os.path.join(shape_path, 'object_unlabeled.mat')
            gt_shape = tf.convert_to_tensor(scipy.io.loadmat(gt_shape_path)['data'], dtype=tf.float32)
            gt_shape = tf.expand_dims(gt_shape, 0)
            gt_shape = tf.expand_dims(gt_shape, 4)
            pred_label = get_pred_label(my_model, gt_shape)
            pred_label = pred_label.numpy().astype('uint8')
            visualization.visualize(pred_label)

    else:

        gt_label = scipy.io.loadmat(os.path.join(single_shape_path, 'object_labeled.mat'))['data']
        visualization.visualize(gt_label)

        gt_shape_path = os.path.join(single_shape_path, 'object_unlabeled.mat')
        gt_shape = tf.convert_to_tensor(scipy.io.loadmat(gt_shape_path)['data'], dtype=tf.float32)
        gt_shape = tf.expand_dims(gt_shape, 0)
        gt_shape = tf.expand_dims(gt_shape, 4)
        pred_label = get_pred_label(my_model, gt_shape)
        pred_label = pred_label.numpy().astype('uint8')
        visualization.visualize(pred_label)


def get_pred_label(model, gt):
    _, composer_output = model(gt)
    pred = tf.squeeze(composer_output[1])
    code = 0
    for idx, each_part in enumerate(pred):
        code += each_part * 2 ** (idx + 1)
    pred_label = tf.math.floor(tf.experimental.numpy.log2(code+1))
    return pred_label


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='this is the script to visualize reconstructed shape')

    parser.add_argument('model_path', help='path of the model')
    parser.add_argument('-m', '--mode', default='batch', help='visualize a batch of shapes or just a single shape. '
                                                              'Valid values are batch and single. Default is batch')
    parser.add_argument('-c', '--category', default='chair', help='which kind of shape to visualize. Default is chair')
    parser.add_argument('-n', '--num_to_visualize', default=4, help='the number of shape to be visualized. Only valid'
                                                                    'when \'mode\' is \'batch\'')
    parser.add_argument('-s', '--single_shape_path', help='path of the shape to be visualized. e.g. '
                                                          'datasets/03001627/1a6f615e8b1b5ae4dbbc9440457e303e. Only '
                                                          'valid when \'mode\' is \'single\'')
    parser.add_argument('-H', default=32, help='height of the shape voxel grid, Default is 32')
    parser.add_argument('-W', default=32, help='width of the shape voxel grid, Default is 32')
    parser.add_argument('-D', default=32, help='depth of the shape voxel grid, Default is 32')
    parser.add_argument('-C', default=1, help='channel of the shape voxel grid, Default is 1')
    args = parser.parse_args()

    evaluate_model(args.model_path, args.mode, args.category, args.num_to_visualize, args.single_shape_path, args.H,
                   args.W, args.D, args.C)
