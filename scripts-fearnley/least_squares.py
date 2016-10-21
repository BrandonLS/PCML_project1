import numpy as np
from costs import compute_loss

def least_squares(y, tx):
    """
	calculate the least squares solution.
    returns mse, and optimal weights
	"""
    w = np.linalg.inv(tx.T.dot(tx)).dot(tx.T).dot(y)
    mse = compute_loss(y,tx,w)
    return mse, w