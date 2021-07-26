import sys
import tensorflow as tf
from utils import dataloader
from datetime import datetime
from hparam import hparam
import model
import os
import json


PROJ_ROOT = os.path.abspath(__file__)[:-8]
RESULT_PATH = os.path.join(PROJ_ROOT, 'results', datetime.now().strftime("%Y%m%d%H%M%S"))


def train_model(category='chair',
                batch_size=32,
                split_ratio=0.8,
                max_num_parts=4,
                optimizer='adam',
                lr=0.001,
                decay_rate=0.8,
                decay_step_size=50,
                decoded_part_threshold=0.125,
                transformed_part_threshold=0.5,
                direct_or_cycle='direct',
                training_process='all',
                epochs=(150, 100, 250),
                model_path=None,
                shuffle=True,
                which_gpu=0):

    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)

    _configure_gpu(which_gpu)
    _save_hparam(hparam, RESULT_PATH)

    # get dataset
    training_set, test_set = dataloader.get_dataset(category=category, batch_size=batch_size, split_ratio=split_ratio,
                                                    max_num_parts=max_num_parts)
    # create model
    my_model = model.Model(max_num_parts, decoded_part_threshold, transformed_part_threshold)

    if training_process == 'all':
        if type(epochs) != tuple:
            raise ValueError('epochs should be a tuple, whose elements inside are the epoch for training '
                             'process 1, 2 and 3 respectively.')
        # training process 1
        _execute_training_process(my_model, training_set, test_set, epochs[0], shuffle, 1, direct_or_cycle, optimizer,
                                  lr, decay_rate, decay_step_size, RESULT_PATH)
        # training process 2
        _execute_training_process(my_model, training_set, test_set, epochs[1], shuffle, 2, direct_or_cycle, optimizer,
                                  lr, decay_rate, decay_step_size, RESULT_PATH)
        # training process 3
        _execute_training_process(my_model, training_set, test_set, epochs[2], shuffle, 3, direct_or_cycle, optimizer,
                                  lr, decay_rate, decay_step_size, RESULT_PATH)

    elif training_process == 1 or training_process == '1':
        if type(epochs) == tuple or type(epochs) == list:
            raise ValueError(f'epochs should be an integer because you only choose process {training_process}')
        _execute_training_process(my_model, training_set, test_set, epochs, shuffle, 1, direct_or_cycle, optimizer, lr,
                                  decay_rate, decay_step_size, RESULT_PATH)

    elif training_process == 2 or training_process == '2':
        if type(epochs) == tuple or type(epochs) == list:
            raise ValueError(f'epochs should be an integer because you only choose process {training_process}')
        warm_up_data = training_set.__iter__().__next__()[0]
        my_model.choose_training_process(2)
        my_model(warm_up_data)
        ans = input(f'Please make sure the model_path is the path of model AFTER training process 1! Your model_path is '
                    f'"{model_path}". Continue? y/n: ')
        while True:
            if ans == 'y' or ans == 'Y' or ans == 'Yes' or ans == 'yes':
                my_model.load_weights(model_path)
                _execute_training_process(my_model, training_set, test_set, epochs, shuffle, 2, direct_or_cycle,
                                          optimizer, lr, decay_rate, decay_step_size, RESULT_PATH)
                break
            elif ans == 'n' or ans == 'N' or ans == 'No' or ans == 'no':
                sys.exit()
            else:
                ans = input('Please enter y/n: ')

    elif training_process == 3 or training_process == '3':
        if type(epochs) == tuple or type(epochs) == list:
            raise ValueError(f'epochs should be an integer because you only choose process {training_process}')
        warm_up_data = training_set.__iter__().__next__()[0]
        my_model.choose_training_process(3)
        my_model(warm_up_data)
        ans = input(f'Please make sure the model_path is the path of model AFTER training process 2! Your model_path is '
                    f'"{model_path}". Continue? y/n: ')
        while True:
            if ans == 'y' or ans == 'Y' or ans == 'Yes' or ans == 'yes':
                my_model.load_weights(model_path)
                _execute_training_process(my_model, training_set, test_set, epochs, shuffle, 3, direct_or_cycle,
                                          optimizer, lr, decay_rate, decay_step_size, RESULT_PATH)
                break
            elif ans == 'n' or ans == 'N' or ans == 'No' or ans == 'no':
                sys.exit()
            else:
                ans = input('Please enter y/n: ')

    else:
        raise ValueError('training_process should be one of \'all\', 1, 2 and 3')


def _configure_gpu(which_gpu):
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_visible_devices(gpus[which_gpu], "GPU")


def _save_hparam(hparam, result_saved_path):
    hparam_saved_path = os.path.join(result_saved_path, 'hparam.json')
    with open(hparam_saved_path, 'w') as f:
        json.dump(hparam, f)


def _get_optimizer(opt, lr):
    if opt == 'adam':
        optimizer = tf.keras.optimizers.Adam(lr)
    elif opt == 'sgd':
        optimizer = tf.keras.optimizers.SGD(lr)
    else:
        raise ValueError(f'{opt} is not valid!')
    return optimizer


def _get_lr_scheduler(decay_rate, decay_step_size):
    def lr_scheduler(epoch, lr):
        if (epoch + 1) % decay_step_size == 0:
            lr = lr * (1 - decay_rate)
        return lr
    return lr_scheduler


def _execute_training_process(my_model,
                              training_set,
                              test_set,
                              epochs,
                              shuffle,
                              process,
                              direct_or_cycle,
                              optimizer,
                              lr,
                              decay_rate,
                              decay_step_size,
                              result_saved_path):
    print(f'Start training process {process}, please wait...')
    process_saved_path = os.path.join(result_saved_path, f'process_{process}')
    if not os.path.exists(process_saved_path):
        os.mkdir(process_saved_path)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(process_saved_path, 'checkpoint.h5'),
                                                             monitor='Transformation_Loss' if process == 2 else 'Total_Loss',
                                                             save_best_only=True,
                                                             save_weights_only=True,
                                                             mode='min')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(process_saved_path, 'logs'),
                                                          histogram_freq=1,
                                                          profile_batch=0)
    lr_scheduler = _get_lr_scheduler(decay_rate, decay_step_size)
    lr_scheduler_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
    callbacks = [checkpoint_callback, tensorboard_callback, lr_scheduler_callback]
    opt = _get_optimizer(optimizer, lr)
    my_model.choose_training_process(training_process=process, mode=direct_or_cycle)
    my_model.compile(optimizer=opt, run_eagerly=True)
    my_model.fit(training_set, epochs=epochs, callbacks=callbacks, shuffle=shuffle)


if __name__ == '__main__':

    ans = input('The training will start soon. Do you want to take some notes for the training? y/n: ')
    while True:
        if ans == 'Y' or ans == 'yes' or ans == 'y':
            if not os.path.exists(RESULT_PATH):
                os.makedirs(RESULT_PATH)
            os.system(f'vi {os.path.join(RESULT_PATH, "training_notes.txt")}')
            print(f'training_notes.txt is saved in {RESULT_PATH}')
            break
        elif ans == 'N' or ans == 'no' or ans == 'n':
            break
        else:
            ans = input('Please enter y/n: ')

    train_model(category=hparam['category'],
                batch_size=hparam['batch_size'],
                split_ratio=hparam['split_ratio'],
                max_num_parts=hparam['max_num_parts'],
                optimizer=hparam['optimizer'],
                lr=hparam['lr'],
                decay_rate=hparam['decay_rate'],
                decay_step_size=hparam['decay_step_size'],
                decoded_part_threshold=hparam['decoded_part_binary_threshold'],
                transformed_part_threshold=hparam['transformed_part_binary_threshold'],
                direct_or_cycle=hparam['direct_or_cycle'],
                training_process=hparam['training_process'],
                epochs=hparam['epochs'],
                model_path=hparam['model_path'],
                shuffle=hparam['shuffle'],
                which_gpu=hparam['which_gpu'])
