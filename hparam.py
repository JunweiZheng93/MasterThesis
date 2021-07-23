hparam = {
    # -------------- dataset relevant ---------------------

    'category': 'chair',
    'max_num_parts': 4,

    # ---------------- training relevant ------------------

    'epochs': (150, 100, 250),  # epochs of training process 1, 2 and 3
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
