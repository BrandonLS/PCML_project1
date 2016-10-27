"""
This file contains all the required ML methods
least_squares_GD
least_squares_SGD
least_squares
ridge_regression
logistic_regression
reg_logistic_regression
"""

#####################################################
#                GRADIENT DESCENT                    #
#####################################################

import numpy as np
from helpers import batch_iter

def compute_loss(y, tx, w):
    """Calculate the mse for weights w"""
    e = y - tx.dot(w)
    return 1/2*np.mean(e**2)

    
def compute_gradient(y, tx, w):
    """Compute the gradient of mse loss function"""
    N = y.shape[0]
    e = y - tx.dot(w)
    return (-1.0/N)*(tx.T).dot(e)

def least_squares_GD(y, tx, initial_w, max_iters, gamma): 
    """
    Linear regression using gradient descent
    @param gamma: step size
    @param max_iters: max number of iterations
    @return: optimal weights, minimum mse
    """
    # Define parameters to store w and loss
    initial_w = np.zeros(tx.shape[1])
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)        
        w = w - gamma * gradient
        # store w and loss
        ws.append(np.copy(w))
        losses.append(loss)
        # print("Gradient Descent({bi}/{ti}): loss={l}".format(
              # bi=n_iter, ti=max_iters - 1, l=loss))

    min_loss = min(losses)
    w = ws[losses.index(min_loss)]
    return w, min_loss

#####################################################
#           STOCHASTIC GRADIENT DESCENT             #
#####################################################
    
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using stochastic gradient descent
    @param gamma: step size
    @param max_iters: maximum number of iterations
    @param batch_size: the size of batchs used for calculating the stochastic gradient
    @return: optimal weights, minimum mse
    """
    batch_size = 5000
    ws = [initial_w]
    losses = []
    w = initial_w
    for i in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            stoch_gradient = compute_gradient(minibatch_y, minibatch_tx, w)
            loss = compute_loss(y, tx, w)        
            w = w - gamma * stoch_gradient
            # store w and loss
            ws.append(np.copy(w))
            losses.append(loss)    
            #print("SGD ({bi}/{ti}): loss={l}".format(bi=i, ti=max_iters - 1, l=loss))            
    min_loss = min(losses)
    w = ws[losses.index(min_loss)]
    return w, min_loss
    

#####################################################
#                   LEAST SQUARES                    #
#####################################################

def least_squares(y, tx):
    """
    Least squares regression using normal equations
    @return:  optimal weights, minimum mse
    """
    w = np.linalg.inv(tx.T.dot(tx)).dot(tx.T).dot(y)
    mse = compute_loss(y,tx,w)
    return w, mse

#####################################################
#               RIDGE REGRESSION                    #
#####################################################

def compute_loss_ridge(y, tx, w, lambda_):
    """
    Calculate loss for ridge regression. Includes a penalising term
    @param lambda_: coefficient for penalising term
    @return: loss for ridge regression
    """
    e = y - tx.dot(w)
    N = y.shape[0]
    return 1.0/(2.0*N)*e.T.dot(e) + lambda_ * np.linalg.norm(w,2)

def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations
    @param lambda_: coefficient for penalinsing term
    @return: optimal weights, minimum mse
    """
    N = y.shape[0]
    w = np.linalg.inv(tx.T.dot(tx) + lambda_*2*N*np.identity(tx.shape[1])).dot(tx.T).dot(y)
    loss = compute_loss_ridge(y,tx,w,lambda_)
    return w, loss
    
    
#####################################################
#               LOGISTIC REGRESSION                 #
#####################################################

def sigmoid(t):
    """apply sigmoid function on t."""
    expo = np.exp(t)
    sigm = expo/(1+expo)
    if np.inf in expo:
        # print("EXP OVERFLOW!!")
        for i, x in enumerate(expo):
            if x == np.inf:
                sigm[i] = 1.0
    return sigm

def calculate_log(y, tx, w):
    """compute the cost by negative log likelihood."""
    loss = sum( np.log(1+np.exp(np.dot(tx,w))) - np.multiply(y,np.dot(tx,w)) )
    # if loss==np.inf:
        # loss = sum( np.dot(tx,w) - np.multiply(y,np.dot(tx,w)) )
        # print("LOSS OVERFLOW!!")
    return loss

def calculate_log_gradient(y, tx, w):
    """compute the gradient of loss."""
    return np.dot(tx.T,(sigmoid(np.dot(tx,w))-y))

def log_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descen using logistic regression.
    @param gamma: learning rate
    @return: the loss and the updated w.
    """
    loss = calculate_log(y, tx, w)
    grad = calculate_log_gradient(y, tx, w)
    w = w - gamma*grad
    return loss, w

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    @param gamma: step size
    @param max_iters: maximum nuber of iterations
    @return : optimal weights, minimum mse
    """
    batch_size = 10000
    losses = []
    w = initial_w
    y_batch = np.zeros((batch_size,1))
    for iter in range(max_iters):
        batch = batch_iter(y, tx, batch_size, num_batches=1, shuffle=True)
        y_batch[:,0],tx_batch = next(batch)
        loss, w = log_gradient_descent(y_batch, tx_batch, w, gamma)
        losses.append(loss)
        # print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))
    return w, loss
    # w = np.zeros((tx.shape[1],1))
    # for iter in range(max_iters):
        # loss, w = log_gradient_descent(y, tx, w, gamma)
        # print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))
    # return loss,w
    
#####################################################
#        REGULARISED LOGISTIC REGRESSION            #
#####################################################

def calculate_reg_log(y, tx, w, lambda_):
    """compute the cost by negative log likelihood."""
    a = np.dot(tx,w)
    return sum( np.log(1+np.exp(a)) - np.multiply(y,a) ) + lambda_ * np.linalg.norm(w,2)

def calculate_reg_log_gradient(y, tx, w, lambda_):
    """compute the gradient of loss."""
    return np.dot(tx.T,(sigmoid(np.dot(tx,w))-y)) + lambda_*2*w

def reg_log_gradient_descent(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descen using logistic regression.
    @param gamma: learning rate
    @return: the loss and the updated w.
    """
    loss = calculate_reg_log(y, tx, w, lambda_)
    grad = calculate_reg_log_gradient(y, tx, w, lambda_)
    w = w - gamma*grad
    return loss, w

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    @param gamma: learning rate
    @param max_iters: maximum nuber of iterations
    @param batch_size: the size of batchs used for calculating the stochastic gradient
    @return : optimal weights, minimum mse
    """
    batch_size = y.shape[0]/10
    losses = []
    w = np.zeros((tx.shape[1],1))
    y_batch = np.zeros((batch_size,1))
    for iter in range(max_iters):
        batch = batch_iter(y, tx, batch_size, num_batches=1, shuffle=True)
        y_batch[:,0],tx_batch = next(batch)
        loss, w = reg_log_gradient_descent(y_batch, tx_batch, w, gamma, lambda_)
        losses.append(loss)
        # print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))
    return w, loss