import tensorflow as tf
import scipy.io
import importlib
import numpy as np
import os
import argparse
from utils import visualization
from copy import deepcopy


PROJ_ROOT = os.path.abspath(__file__)[:-11]
CATEGORY_MAP = {'chair': '03001627', 'table': '04379243', 'airplane': '02691156', 'lamp': '03636649'}


def evaluate_model(model_path,
                   mode='batch',
                   category='chair',
                   num_to_visualize=4,
                   single_shape_path=None,
                   exchange_shape_path=None,
                   which_part=1,
                   H=32,
                   W=32,
                   D=32,
                   C=1,
                   visualize_decoded_part=False,
                   decoded_part_threshold=0.125,
                   transformed_part_threshold=0.5):

    # check category
    if category not in ['chair', 'table', 'airplane', 'lamp']:
        raise ValueError('category should be one of chair, table, airplane and lamp!')

    # check mode
    if mode == 'batch':
        num_parts = 3 if category == 'table' else 4
    elif mode == 'single':
        if not single_shape_path.endswith('/'):
            single_shape_path += '/'
        category_code = single_shape_path.split('/')[-3]
        category = list(CATEGORY_MAP.keys())[list(CATEGORY_MAP.values()).index(category_code)]
        num_parts = 3 if category == 'table' else 4
    elif mode == 'exchange':
        if type(exchange_shape_path) != list:
            raise ValueError(f'exchange_shape_path should be a list. got {type(exchange_shape_path)}')
        num_path = len(exchange_shape_path)
        if num_path != 2:
            raise ValueError(f'you should input 2 paths for the exchange. got {num_path} instead!')
        path_1 = exchange_shape_path[0]
        path_2 = exchange_shape_path[1]
        if not path_1.endswith('/'):
            path_1 += '/'
        if not path_2.endswith('/'):
            path_2 += '/'
        exchange_shape_path = [path_1, path_2]
        path_1_category_code = path_1.split('/')[-3]
        path_2_category_code = path_2.split('/')[-3]
        if path_1_category_code != path_2_category_code:
            raise ValueError(f'you should input 2 shape paths from the same category. '
                             f'got {path_1}({list(CATEGORY_MAP.keys())[list(CATEGORY_MAP.values()).index(path_1)]}) '
                             f'and {path_2}({list(CATEGORY_MAP.keys())[list(CATEGORY_MAP.values()).index(path_2)]})')
        category = list(CATEGORY_MAP.keys())[list(CATEGORY_MAP.values()).index(path_1_category_code)]
        num_parts = 3 if category == 'table' else 4
    elif mode == 'assembly':
        pass
    else:
        raise ValueError('mode should be one of batch, single, exchange and assembly!')

    # load weights and warm up model
    warm_up_data = tf.ones((1, H, W, D, C), dtype=tf.float32)
    model = importlib.import_module(f"results.{model_path.split('/')[-3]}.model")
    my_model = model.Model(num_parts=num_parts)
    my_model.choose_training_process(1)
    my_model(warm_up_data)
    my_model.load_weights(model_path)

    dataset_path = os.path.join(PROJ_ROOT, 'datasets', CATEGORY_MAP[category])

    if mode == 'batch':
        all_shapes = os.listdir(dataset_path)
        idx = np.random.choice(len(all_shapes), num_to_visualize, replace=False)
        shapes_to_visualize = [all_shapes[each] for each in idx]

        for shape_code in shapes_to_visualize:
            # visualize ground truth label
            gt_label_path = os.path.join(dataset_path, shape_code, 'object_labeled.mat')
            gt_label = scipy.io.loadmat(gt_label_path)['data']
            visualization.visualize(gt_label, title=shape_code)

            shape_path = os.path.dirname(gt_label_path)
            gt_shape_path = os.path.join(shape_path, 'object_unlabeled.mat')
            _visualize_pred_label(my_model, 'batch', gt_shape_path, shape_code, visualize_decoded_part,
                                  decoded_part_threshold, transformed_part_threshold, which_part)

    elif mode == 'single':
        # visualize ground truth label
        shape_code = single_shape_path.split('/')[-2]
        gt_label = scipy.io.loadmat(os.path.join(single_shape_path, 'object_labeled.mat'))['data']
        visualization.visualize(gt_label, title=shape_code)

        gt_shape_path = os.path.join(single_shape_path, 'object_unlabeled.mat')
        _visualize_pred_label(my_model, 'single', gt_shape_path, shape_code, visualize_decoded_part,
                              decoded_part_threshold, transformed_part_threshold, which_part)

    elif mode == 'exchange':
        # visualize ground truth label
        shape_code_1 = exchange_shape_path[0].split('/')[-2]
        shape_code_2 = exchange_shape_path[1].split('/')[-2]
        gt_label_1 = scipy.io.loadmat(os.path.join(exchange_shape_path[0], 'object_labeled.mat'))['data']
        gt_label_2 = scipy.io.loadmat(os.path.join(exchange_shape_path[1], 'object_labeled.mat'))['data']
        visualization.visualize(gt_label_1, title=shape_code_1)
        visualization.visualize(gt_label_2, title=shape_code_2)

        # get latent representation
        gt_shape_path_1 = os.path.join(exchange_shape_path[0], 'object_unlabeled.mat')
        gt_shape_path_2 = os.path.join(exchange_shape_path[1], 'object_unlabeled.mat')
        _visualize_pred_label(my_model, 'exchange', (gt_shape_path_1, gt_shape_path_2), (shape_code_1, shape_code_2),
                              visualize_decoded_part, decoded_part_threshold, transformed_part_threshold, which_part)


def _visualize_pred_label(model,
                          mode,
                          shape_path,
                          shape_code,
                          visualize_decoded_part=False,
                          decoded_part_threshold=0.125,
                          transformed_part_threshold=0.5,
                          which_part=1):
    """
    :param model: tensorflow model
    :param mode: valid values are batch, single, exchange and assembly
    :param shape_path: it should be a tuple when mode is 'exchange', should be a str for other modes
    :param shape_code: it should be a tuple when mode is 'exchange', should be a str for other modes
    :param visualize_decoded_part: whether to visualize decoded parts
    :param decoded_part_threshold: threshold for decoded parts to be visualized
    :param transformed_part_threshold: threshold for transformed parts to be visualized
    :param which_part: which part to be exchange. only valid when mode is 'exchange'
    """
    if mode == 'exchange':
        gt_shape_path_1 = shape_path[0]
        gt_shape_path_2 = shape_path[1]
        gt_shape_1 = tf.convert_to_tensor(scipy.io.loadmat(gt_shape_path_1)['data'], dtype=tf.float32)
        gt_shape_2 = tf.convert_to_tensor(scipy.io.loadmat(gt_shape_path_2)['data'], dtype=tf.float32)
        gt_shape_1 = tf.expand_dims(gt_shape_1, 0)
        gt_shape_2 = tf.expand_dims(gt_shape_2, 0)
        gt_shape_1 = tf.expand_dims(gt_shape_1, 4)
        gt_shape_2 = tf.expand_dims(gt_shape_2, 4)
        latent_1 = model.decomposer(gt_shape_1)
        latent_2 = model.decomposer(gt_shape_2)

        def _visualize_exchange_pred_label(latent_1, latent_2, shape_code):
            model.composer(latent_1)
            pred_1 = tf.squeeze(tf.where(model.composer.stn_output_fmap > transformed_part_threshold, 1., 0.))
            model.composer(latent_2)
            pred_2 = tf.squeeze(tf.where(model.composer.stn_output_fmap > transformed_part_threshold, 1., 0.))
            code_1 = 0
            code_2 = 0
            for idx, each_part in enumerate(pred_1):
                code_1 += each_part * 2 ** (idx + 1)
            for idx, each_part in enumerate(pred_2):
                code_2 += each_part * 2 ** (idx + 1)
            pred_label_1 = tf.math.floor(tf.experimental.numpy.log2(code_1 + 1))
            pred_label_2 = tf.math.floor(tf.experimental.numpy.log2(code_2 + 1))
            pred_label_1 = pred_label_1.numpy().astype('uint8')
            pred_label_2 = pred_label_2.numpy().astype('uint8')
            visualization.visualize(pred_label_1, title=shape_code[0])
            visualization.visualize(pred_label_2, title=shape_code[1])

        # visualize pred_label before exchange
        _visualize_exchange_pred_label(latent_1, latent_2, shape_code)

        # exchange part
        latent_array_1 = latent_1.numpy()
        latent_array_2 = latent_2.numpy()
        temp = deepcopy(latent_array_1[:, which_part-1, :])
        latent_array_1[:, which_part-1, :] = latent_array_2[:, which_part-1, :]
        latent_array_2[:, which_part-1, :] = temp
        latent_1 = tf.convert_to_tensor(latent_array_1)
        latent_2 = tf.convert_to_tensor(latent_array_2)

        # visualize pred_label after exchange
        _visualize_exchange_pred_label(latent_1, latent_2, shape_code)

    else:
        gt_shape = tf.convert_to_tensor(scipy.io.loadmat(shape_path)['data'], dtype=tf.float32)
        gt_shape = tf.expand_dims(gt_shape, 0)
        gt_shape = tf.expand_dims(gt_shape, 4)
        model(gt_shape)
        if visualize_decoded_part:
            pred = tf.squeeze(tf.where(model.composer.stacked_decoded_parts > decoded_part_threshold, 1., 0.))
            pred = pred.numpy().astype('uint8')
            for part in pred:
                visualization.visualize(part, title=shape_code)
        else:
            pred = tf.squeeze(tf.where(model.composer.stn_output_fmap > transformed_part_threshold, 1., 0.))
            code = 0
            for idx, each_part in enumerate(pred):
                code += each_part * 2 ** (idx + 1)
            pred_label = tf.math.floor(tf.experimental.numpy.log2(code+1))
            pred_label = pred_label.numpy().astype('uint8')
            visualization.visualize(pred_label, title=shape_code)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='this is the script to visualize reconstructed shape')

    parser.add_argument('model_path', help='path of the model')
    parser.add_argument('-m', '--mode', default='batch', help='visualize a batch of shapes or just a single shape. '
                                                              'Valid values are batch, single, exchange and assembly. '
                                                              'Default is batch')
    parser.add_argument('-c', '--category', default='chair', help='which kind of shape to visualize. Default is chair')
    parser.add_argument('-n', '--num_to_visualize', default=4, help='the number of shape to be visualized. Only valid'
                                                                    'when \'mode\' is \'batch\'')
    parser.add_argument('-s', '--single_shape_path', default=None, help='path of the shape to be visualized. e.g. '
                                                          'datasets/03001627/1a6f615e8b1b5ae4dbbc9440457e303e. Only '
                                                          'valid when \'mode\' is \'single\'')
    parser.add_argument('-e', '--exchange_shape_path', nargs='*', help='2 shape paths used to exchange parts')
    parser.add_argument('-w', '--which_part', default=1, help='which part to be exchanged. Only valid when mode is \'exchange\'')
    parser.add_argument('-H', default=32, help='height of the shape voxel grid. Default is 32')
    parser.add_argument('-W', default=32, help='width of the shape voxel grid. Default is 32')
    parser.add_argument('-D', default=32, help='depth of the shape voxel grid. Default is 32')
    parser.add_argument('-C', default=1, help='channel of the shape voxel grid. Default is 1')
    parser.add_argument('-v', '--visualize_decoded_part', action="store_true", help='whether to visualize decoded parts')
    parser.add_argument('-d', '--decoded_part_threshold', default=0.125, help='threshold of decoded parts to be visualized. '
                                                                              'Default is 0.125')
    parser.add_argument('-t', '--transformed_part_threshold', default=0.5, help='threshold of transformed parts to be visualized')
    args = parser.parse_args()

    evaluate_model(model_path=args.model_path,
                   mode=args.mode,
                   category=args.category,
                   num_to_visualize=int(args.num_to_visualize),
                   single_shape_path=args.single_shape_path,
                   exchange_shape_path=args.exchange_shape_path,
                   which_part=int(args.which_part),
                   H=int(args.H),
                   W=int(args.W),
                   D=int(args.D),
                   C=int(args.C),
                   visualize_decoded_part=args.visualize_decoded_part,
                   decoded_part_threshold=float(args.decoded_part_threshold),
                   transformed_part_threshold=float(args.transformed_part_threshold))
