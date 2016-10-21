# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np


def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    training_len = int(ratio * len(x))
    test_len = len(x) - training_len
    p = np.random.permutation(x.shape[0])
    shuffled_x = x[p]
    shuffled_y = y[p]

	
	
    train_x, train_y = shuffled_x[:training_len,], shuffled_y[:training_len,]
    test_x, test_y = shuffled_x[:test_len,], shuffled_y[:test_len,] 
    return train_x, train_y, test_x, test_y