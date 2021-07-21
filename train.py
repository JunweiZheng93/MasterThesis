import tensorflow as tf
from utils import dataloader
from datetime import datetime
import model
import argparse
import os


PROJ_ROOT = os.path.abspath(__file__)[:-8]


def train_model(category='chair',
                batch_size=32,
                split_ratio=0.8,
                max_num_parts=4,
                optimizer='adam',
                epochs=(150, 100, 250),
                shuffle=True,
                info_verbose=0,
                resume_training=False,
                resume_training_path=None,
                initial_epoch=0,
                which_gpu=0):

    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_visible_devices(gpus[which_gpu], "GPU")

    training_set, test_set = dataloader.get_dataset(category=category, batch_size=batch_size, split_ratio=split_ratio,
                                                    max_num_parts=max_num_parts)
    result_saved_path = os.path.join(PROJ_ROOT, 'results', datetime.now().strftime("%Y%m%d%H%M%S"))
    my_model = model.Model(num_parts=max_num_parts)

    # training process 1
    print('Start training process 1, please wait...')
    process1_saved_path = os.path.join(result_saved_path, 'process_1')
    if not os.path.exists(process1_saved_path):
        os.makedirs(process1_saved_path)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(process1_saved_path, 'checkpoint.h5'),
                                                             monitor='Total_Loss',
                                                             verbose=info_verbose,
                                                             save_best_only=True,
                                                             save_weights_only=True,
                                                             mode='min')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(process1_saved_path, 'logs'),
                                                          histogram_freq=1,
                                                          profile_batch=0)
    my_model.compile(optimizer=optimizer, run_eagerly=True)
    callbacks = [checkpoint_callback, tensorboard_callback]
    my_model.choose_training_process(training_process=1)
    my_model.fit(training_set, epochs=epochs[0], callbacks=callbacks, shuffle=shuffle, initial_epoch=initial_epoch)

    # training process 2
    print('Start training process 2, please wait...')
    process2_saved_path = os.path.join(result_saved_path, 'process_2')
    if not os.path.exists(process2_saved_path):
        os.makedirs(process2_saved_path)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(process2_saved_path, 'checkpoint.h5'),
                                                             monitor='Transformation_Loss',
                                                             verbose=info_verbose,
                                                             save_best_only=True,
                                                             save_weights_only=True,
                                                             mode='min')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(process2_saved_path, 'logs'),
                                                          histogram_freq=1,
                                                          profile_batch=0)
    my_model.compile(optimizer=optimizer, run_eagerly=True)
    callbacks = [checkpoint_callback, tensorboard_callback]
    my_model.choose_training_process(training_process=2)
    my_model.fit(training_set, epochs=epochs[1], callbacks=callbacks, shuffle=shuffle, initial_epoch=initial_epoch)

    # training process 3
    print('Start training process 3, please wait...')
    process3_saved_path = os.path.join(result_saved_path, 'process_3')
    if not os.path.exists(process3_saved_path):
        os.makedirs(process3_saved_path)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(process3_saved_path, 'checkpoint.h5'),
                                                             monitor='Total_Loss',
                                                             verbose=info_verbose,
                                                             save_best_only=True,
                                                             save_weights_only=True,
                                                             mode='min')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(process3_saved_path, 'logs'),
                                                          histogram_freq=1,
                                                          profile_batch=0)
    my_model.compile(optimizer=optimizer, run_eagerly=True)
    callbacks = [checkpoint_callback, tensorboard_callback]
    my_model.choose_training_process(training_process=3)
    my_model.fit(training_set, epochs=epochs[2], callbacks=callbacks, shuffle=shuffle, initial_epoch=initial_epoch)


if __name__ == '__main__':

    train_model(category='chair',
                batch_size=32,
                split_ratio=0.8,
                max_num_parts=4,
                optimizer='adam',
                epochs=(10, 6, 15),
                shuffle=True,
                info_verbose=0,
                resume_training=False,
                resume_training_path=None,
                initial_epoch=0,
                which_gpu=7)
