# -*- coding: utf-8 -*-
"""
Stochastic Gradient Descent
"""

import numpy as np


def compute_loss_ridge(y, tx, w, lamb):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    # MSE
    e = y - tx.dot(w)
    N = y.shape[0]
    return 1.0/(2.0*N)*e.T.dot(e) + lamb * np.linalg.norm(w,2)

def ridge_regression(y, tx, lamb):
    """implement ridge regression."""
    N = y.shape[0]
    w = np.linalg.inv(tx.T.dot(tx) + lamb*2*N*np.identity(tx.shape[1])).dot(tx.T).dot(y)
    loss = compute_loss_ridge(y,tx,w,lamb)
    return loss, w