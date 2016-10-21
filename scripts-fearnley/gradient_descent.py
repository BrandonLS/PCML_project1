# -*- coding: utf-8 -*-
"""Problem Sheet 2.

Gradient Descent
"""
import numpy as np
from costs import *

def compute_gradient_mse(y, tx, w):
    """
    Compute the gradient.
    For mse with linear model
    """
    N = y.shape[0]
    e = y - tx.dot(w)
    return (-1.0/N)*(tx.T).dot(e)
    
def compute_gradient_mae(y, tx, w):
    """Compute the gradient for mae"""
    N, D = tx.shape[0], tx.shape[1]
    e = y - tx.dot(w)
    c = np.sign(e)
    
    grad = np.zeros(D)
    # grad[0] = -1.0/N * np.sum(c)
    for i in range(D):
        grad[i] = -1.0/N * tx[:,i].T.dot(c)
    return grad
	
def compute_gradient(y, tx, w, loss_func='mse'):
	if loss_func == 'mse':
		return compute_gradient_mse(y, tx, w)
	elif loss_func == 'mae':
		return compute_gradient_mae(y, tx, w)
	else:
		raise Exception('Invalid loss function.')

def gradient_descent(y, tx, initial_w, max_iters, gamma, loss_func='mse'): 
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w, loss_func=loss_func)
        loss = compute_loss(y, tx, w)        
        w = w - gamma * gradient
        # store w and loss
        ws.append(np.copy(w))
        losses.append(loss)
        # print("Gradient Descent({bi}/{ti}): loss={l}".format(
              # bi=n_iter, ti=max_iters - 1, l=loss))

    min_loss = min(losses)
    w = ws[losses.index(min_loss)]
    return min_loss, w
