import os
import numpy as np
import tensorflow as tf
import math
import scipy.io
from tensorflow.keras.utils import Sequence
from tqdm import tqdm

CATEGORY_MAP = {'chair': '03001627', 'table': '04379243', 'airplane': '02691156', 'lamp': '03636649'}
URL_MAP = {'chair': 'https://gitlab.com/JunweiZheng93/shapenetsegvox/-/raw/master/03001627.zip?inline=false',
           'table': '',
           'airplane': 'https://gitlab.com/JunweiZheng93/shapenetsegvox/-/raw/master/02691156.zip?inline=false',
           'lamp': 'https://gitlab.com/JunweiZheng93/shapenetsegvox/-/raw/master/03636649.zip?inline=false'}
PROJ_ROOT = os.path.abspath(__file__)[:-19]


# def get_dataset(category='chair', batch_size=32, split_ratio=0.8, max_num_parts=4):
#
#     category_path = download_dataset(category)
#     voxel_grid_fp, part_fp, trans_fp = get_fp(category_path)
#     num_training_samples = math.ceil(len(voxel_grid_fp) * split_ratio)
#     training_set = Dataset(voxel_grid_fp[:num_training_samples], part_fp[:num_training_samples],
#                            trans_fp[:num_training_samples], batch_size, max_num_parts)
#     test_set = Dataset(voxel_grid_fp[num_training_samples:], part_fp[num_training_samples:],
#                        trans_fp[num_training_samples:], batch_size, max_num_parts)
#
#     return training_set, test_set

def get_dataset(category='chair', batch_size=32, split_ratio=0.8, max_num_parts=4):

    category_path = download_dataset(category)
    voxel_grid_fp, part_fp, trans_fp = get_fp(category_path)
    num_training_samples = math.ceil(len(voxel_grid_fp) * split_ratio)

    all_voxel_grid = list()
    all_part = list()
    all_trans = list()
    print('Loading data to RAM. Please wait...')
    for v_fp, p_fp, t_fp in tqdm(zip(voxel_grid_fp, part_fp, trans_fp), total=len(voxel_grid_fp)):
        v = scipy.io.loadmat(v_fp)['data'][:, :, :, np.newaxis]
        all_voxel_grid.append(v)

        parts = list()
        transformations = list()
        member_list = [int(each[-5]) for each in p_fp]
        dir_name = os.path.dirname(v_fp)
        for i in range(1, max_num_parts + 1):
            if i not in member_list:
                part = np.zeros_like(v, dtype='uint8')
                parts.append(part)
                transformation = np.zeros((3, 4), dtype='float32')
                transformations.append(transformation)
            else:
                part = scipy.io.loadmat(os.path.join(dir_name, f'part{i}.mat'))['data'][:, :, :, np.newaxis]
                parts.append(part)
                transformations.append(scipy.io.loadmat(os.path.join(dir_name, f'part{i}_trans_matrix.mat'))['data'][:3])
        all_part.append(parts)
        all_trans.append(transformations)

    training_set = Dataset(all_voxel_grid[:num_training_samples], all_part[:num_training_samples],
                           all_trans[:num_training_samples], batch_size, max_num_parts)
    test_set = Dataset(all_voxel_grid[num_training_samples:], all_part[num_training_samples:],
                       all_trans[num_training_samples:], batch_size, max_num_parts)

    return training_set, test_set

# def get_dataset(category='chair', batch_size=32, split_ratio=0.8, max_num_parts=4):
#
#     category_path = download_dataset(category)
#     voxel_grid_fp, part_fp, trans_fp = get_fp(category_path)
#     num_training_samples = math.ceil(len(voxel_grid_fp) * split_ratio)
#     parse_function = ParseFunction(max_num_parts)
#
#     training_gen = Gen(voxel_grid_fp[:num_training_samples], part_fp[:num_training_samples], trans_fp[:num_training_samples])
#     training_set = tf.data.Dataset.from_generator(training_gen, (tf.string, tf.string, tf.string))
#     training_set = training_set.map(lambda voxel_grid_fp, part_fp, trans_fp: tf.py_function(func=parse_function, inp=[voxel_grid_fp, part_fp, trans_fp], Tout=[tf.float32, tf.float32, tf.float32]),
#                                     num_parallel_calls=tf.data.experimental.AUTOTUNE)
#     training_set = training_set.cache()
#     training_set = training_set.shuffle(len(voxel_grid_fp[:num_training_samples]), seed=0, reshuffle_each_iteration=True)
#     training_set = training_set.batch(batch_size, drop_remainder=False)
#     training_set = training_set.prefetch(tf.data.experimental.AUTOTUNE)
#
#     test_gen = Gen(voxel_grid_fp[num_training_samples:], part_fp[num_training_samples:], trans_fp[num_training_samples:])
#     test_set = tf.data.Dataset.from_generator(test_gen, (tf.string, tf.string, tf.string))
#     test_set = test_set.map(lambda voxel_grid_fp, part_fp, trans_fp: tf.py_function(func=parse_function, inp=[voxel_grid_fp, part_fp, trans_fp], Tout=[tf.float32, tf.float32, tf.float32]),
#                             num_parallel_calls=tf.data.experimental.AUTOTUNE)
#     test_set = test_set.cache()
#     test_set = test_set.shuffle(len(voxel_grid_fp[num_training_samples:]), seed=0, reshuffle_each_iteration=True)
#     test_set = test_set.batch(batch_size, drop_remainder=False)
#     test_set = test_set.prefetch(tf.data.experimental.AUTOTUNE)
#
#     return training_set, test_set


def download_dataset(category):

    # check category
    if category not in list(CATEGORY_MAP.keys()):
        raise ValueError(f'category should be one of chair, table, airplane and lamp. got {category} instead!')

    category_path = os.path.join(PROJ_ROOT, 'datasets', CATEGORY_MAP[category])
    if not os.path.exists(category_path):
        tf.keras.utils.get_file(f'{CATEGORY_MAP[category]}.zip', URL_MAP[category], cache_dir=PROJ_ROOT, extract=True)
        os.remove(f'{category_path}.zip')
    return category_path


def get_fp(category_fp):
    shape_paths = [os.path.join(category_fp, shape_name) for shape_name in os.listdir(category_fp)]
    voxel_grid_fp = list()
    part_fp = list()
    trans_fp = list()
    for shape_path in shape_paths:
        voxel_grid = os.path.join(shape_path, 'object_unlabeled.mat')
        part_list = list()
        trans_list = list()
        all_files = os.listdir(shape_path)
        for file in all_files:
            if file.startswith('part') and file.endswith('.mat'):
                if file.startswith('part') and file.endswith('trans_matrix.mat'):
                    trans_list.append(os.path.join(shape_path, file))
                    continue
                part_list.append(os.path.join(shape_path, file))
        voxel_grid_fp.append(voxel_grid)
        part_fp.append(part_list)
        trans_fp.append(trans_list)
    return voxel_grid_fp, part_fp, trans_fp


# class Gen:
#     def __init__(self, voxel_grid_fp, part_fp, trans_fp):
#         self.voxel_grid_fp = voxel_grid_fp
#         self.part_fp = part_fp
#         self.trans_fp = trans_fp
#
#     def __call__(self, *args, **kwargs):
#         for voxel_grid, part, trans in zip(self.voxel_grid_fp, self.part_fp, self.trans_fp):
#             yield tf.convert_to_tensor(voxel_grid), tf.convert_to_tensor(part), tf.convert_to_tensor(trans)


# class ParseFunction:
#
#     def __init__(self, max_num_parts):
#         self.max_num_parts = max_num_parts
#
#     def __call__(self, voxel_grid_fp, part_fp, trans_fp):
#         voxel_grid_fp = voxel_grid_fp.numpy()
#         part_fp = part_fp.numpy()
#
#         voxel_grid = scipy.io.loadmat(voxel_grid_fp)['data'][:, :, :, np.newaxis]
#         parts = list()
#         transformations = list()
#         member_list = [int(each[-5]) for each in part_fp]
#         dir_name = os.path.dirname(voxel_grid_fp)
#         for i in range(1, self.max_num_parts+1):
#             if i not in member_list:
#                 part = np.zeros_like(voxel_grid, dtype='uint8')
#                 parts.append(part)
#                 transformation = np.zeros((3, 4), dtype='float32')
#                 transformations.append(transformation)
#             else:
#                 part = scipy.io.loadmat(os.path.join(dir_name, f'part{i}.mat'))['data'][:, :, :, np.newaxis]
#                 parts.append(part)
#                 transformations.append(scipy.io.loadmat(os.path.join(dir_name, f'part{i}_trans_matrix.mat'))['data'][:3])
#         return tf.convert_to_tensor(voxel_grid, dtype=tf.float32), tf.convert_to_tensor(parts, dtype=tf.float32), tf.convert_to_tensor(transformations, dtype=tf.float32)


# class Dataset(Sequence):
#
#     def __init__(self, voxel_grid_fp, part_fp, trans_fp, batch_size, max_num_parts):
#         self.voxel_grid_fp = voxel_grid_fp
#         self.part_fp = part_fp
#         self.trans_fp = trans_fp
#         self.batch_size = batch_size
#         self.max_num_parts = max_num_parts
#
#     def __len__(self):
#         return math.ceil(len(self.voxel_grid_fp) / self.batch_size)
#
#     def __getitem__(self, idx):
#         batch_voxel_grid_fp = self.voxel_grid_fp[idx * self.batch_size: (idx + 1) * self.batch_size]
#         batch_part_fp = self.part_fp[idx * self.batch_size: (idx + 1) * self.batch_size]
#         batch_trans_fp = self.trans_fp[idx * self.batch_size: (idx + 1) * self.batch_size]
#
#         batch_voxel_grid = list()
#         batch_part = list()
#         batch_trans = list()
#         for voxel_grid_fp, part_fp, trans_fp in zip(batch_voxel_grid_fp, batch_part_fp, batch_trans_fp):
#             voxel_grid = scipy.io.loadmat(voxel_grid_fp)['data'][:, :, :, np.newaxis]
#             batch_voxel_grid.append(voxel_grid)
#
#             parts = list()
#             transformations = list()
#             member_list = [int(each[-5]) for each in part_fp]
#             dir_name = os.path.dirname(voxel_grid_fp)
#             for i in range(1, self.max_num_parts+1):
#                 if i not in member_list:
#                     part = np.zeros_like(voxel_grid, dtype='uint8')
#                     parts.append(part)
#                     transformation = np.zeros((3, 4), dtype='float32')
#                     transformations.append(transformation)
#                 else:
#                     part = scipy.io.loadmat(os.path.join(dir_name, f'part{i}.mat'))['data'][:, :, :, np.newaxis]
#                     parts.append(part)
#                     transformations.append(scipy.io.loadmat(os.path.join(dir_name, f'part{i}_trans_matrix.mat'))['data'][:3])
#             batch_part.append(parts)
#             batch_trans.append(transformations)
#
#         return np.asarray(batch_voxel_grid, dtype='float32'), np.asarray(batch_part, dtype='float32'), np.asarray(batch_trans, dtype='float32')


class Dataset(Sequence):

    def __init__(self, all_voxel_grid, all_part, all_trans, batch_size, max_num_parts):
        self.all_voxel_grid = all_voxel_grid
        self.all_part = all_part
        self.all_trans = all_trans
        self.batch_size = batch_size
        self.max_num_parts = max_num_parts

    def __len__(self):
        return math.ceil(len(self.all_voxel_grid) / self.batch_size)

    def __getitem__(self, idx):
        batch_voxel_grid = self.all_voxel_grid[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_part = self.all_part[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_trans = self.all_trans[idx * self.batch_size: (idx + 1) * self.batch_size]
        return np.asarray(batch_voxel_grid, dtype='float32'), np.asarray(batch_part, dtype='float32'), np.asarray(batch_trans, dtype='float32')