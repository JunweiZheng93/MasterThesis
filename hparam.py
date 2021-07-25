hparam = {
    # -------------- dataset relevant ---------------------

    'category': 'chair',
    'max_num_parts': 4,

    # ---------------- training relevant ------------------

    'training_process': 'all',  # should be one of 'all', 1, 2 and 3
    'model_path': None,  # only valid when training_process is 2 or 3. If training_process is 2, model_path should
                         # be the path of model after training process 1. If training_process is 3, model_path should
                         # be the path of model after training process 2
    'epochs': (150, 100, 250),  # it should be a tuple, whose element inside are epoch for training process 1, 2
                                # and 3 if training_process is 'all'; otherwise it's an integer and only for one
                                # specific training process
    'batch_size': 32,
    'split_ratio': 0.8,  # ratio to split dataset into training set and test set
    'shuffle': True,
    'optimizer': 'adam',  # adam or sgd
    'lr': 0.001,
    'decay_rate': 0.8,  # decay rate of learning rate
    'decay_step_size': 50,

    # ---------------- other settings --------------------

    'which_gpu': 7,
}
