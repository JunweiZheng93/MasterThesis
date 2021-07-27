hparam = {
    # -------------- dataset relevant ---------------------

    'category': 'chair',
    'max_num_parts': 4,

    # ---------------- training relevant ------------------

    'training_process': '3',  # should be one of 'all', 1, 2 and 3

    # only valid when training_process is 2 or 3. If training_process is 2, model_path should
    # be the path of model after training process 1. If training_process is 3, model_path should
    # be the path of model after training process 2
    'model_path': '/home/ies/zhen/MasterThesis/results/20210726105557/process_2/checkpoint.h5',

    # epochs should be a tuple, whose element inside are epoch for training process 1, 2
    # and 3 if training_process is 'all'; otherwise it's an integer and only for one
    # specific training process
    'epochs': 250,
    'batch_size': 64,
    'split_ratio': 0.8,  # ratio to split dataset into training set and test set
    'shuffle': True,
    'optimizer': 'adam',  # adam or sgd
    'lr': 0.0001,
    'decay_rate': 0.8,  # decay rate of learning rate
    'decay_step_size': 50,
    'decoded_part_binary_threshold': 0.125,  # threshold for the binary decoded part
    'transformed_part_binary_threshold': 0.5,  # threshold for the binary transformed part

    # use shape reconstruction loss or cycle loss for training process 3 (only valid for training process 3)
    'direct_or_cycle': 'direct',

    # ---------------- other settings --------------------

    'which_gpu': 7,
}
