# -*- coding: utf-8 -*-
"""
Stochastic Gradient Descent
"""
from helpers import batch_iter
from gradient_descent import compute_gradient
from costs import compute_loss
import numpy as np

def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_epochs, gamma, loss_func='mse', coef=0.0):
    """Stochastic gradient descent algorithm.
    @return: min_loss, w the minimum value of mse and the w for which it is attained
    """
    
    ws = [initial_w]
    losses = []
    w = initial_w
    count = 0
    for i in range(max_epochs):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            stoch_gradient = compute_gradient(minibatch_y, minibatch_tx, w, loss_func=loss_func)
            loss = compute_loss(y, tx, w)        
            w = w - gamma * (1.0-coef)**count * stoch_gradient
            # store w and loss
            ws.append(np.copy(w))
            losses.append(loss)    
            count += 1
            #print("SGD ({bi}/{ti}): loss={l}".format(bi=i, ti=max_epochs - 1, l=loss))            
    min_loss = min(losses)
    w = ws[losses.index(min_loss)]
    return min_loss, w
