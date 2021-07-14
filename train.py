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
                run_eagerly=True,
                epochs=(150, 100, 250),
                shuffle=True,
                max_queue_size=32,
                workers=4,
                use_multiprocessing=True,
                info_verbose=0,
                resume_training=False,
                resume_training_path=None,
                initial_epoch=0):

    training_set, test_set = dataloader.get_dataset(category=category, batch_size=batch_size, split_ratio=split_ratio,
                                                    max_num_parts=max_num_parts)
    result_saved_path = os.path.join(PROJ_ROOT, 'results', datetime.now().strftime("%Y%m%d%H%M%S"))
    my_model = model.Model(num_parts=max_num_parts)

    # training process 1
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
    my_model.compile(optimizer=optimizer, run_eagerly=run_eagerly)
    callbacks = [checkpoint_callback, tensorboard_callback]
    my_model.choose_training_process(training_process=1)
    my_model.fit(training_set, epochs=epochs[0], callbacks=callbacks, validation_data=test_set, shuffle=shuffle,
                 initial_epoch=initial_epoch, max_queue_size=max_queue_size, workers=workers,
                 use_multiprocessing=use_multiprocessing)

    # training process 2
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
    my_model.compile(optimizer=optimizer, run_eagerly=run_eagerly)
    callbacks = [checkpoint_callback, tensorboard_callback]
    my_model.choose_training_process(training_process=2)
    my_model.fit(training_set, epochs=epochs[1], callbacks=callbacks, validation_data=test_set, shuffle=shuffle,
                 initial_epoch=initial_epoch, max_queue_size=max_queue_size, workers=workers,
                 use_multiprocessing=use_multiprocessing)

    # training process 3
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
    my_model.compile(optimizer=optimizer, run_eagerly=run_eagerly)
    callbacks = [checkpoint_callback, tensorboard_callback]
    my_model.choose_training_process(training_process=3)
    my_model.fit(training_set, epochs=epochs[2], callbacks=callbacks, validation_data=test_set, shuffle=shuffle,
                 initial_epoch=initial_epoch, max_queue_size=max_queue_size, workers=workers,
                 use_multiprocessing=use_multiprocessing)


if __name__ == '__main__':

    train_model(category='chair',
                batch_size=1,
                split_ratio=0.5,
                max_num_parts=4,
                optimizer='adam',
                run_eagerly=True,
                epochs=(2, 2, 2),
                shuffle=True,
                max_queue_size=32,
                workers=4,
                use_multiprocessing=True,
                info_verbose=0,
                resume_training=False,
                resume_training_path=None,
                initial_epoch=0)
