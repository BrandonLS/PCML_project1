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
#                GRADIENT DESCENT					#
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

def least_squares_GD(y, tx, gamma, max_iters): 
    """
	Linear regression using gradient descent
	@param gamma: learning rate
	@param max_iters: max number of iterations
	@return: an approximation of the minimum of the mse loss function and the associated weights
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
    return min_loss, w

#####################################################
#           STOCHASTIC GRADIENT DESCENT				#
#####################################################
	
def least_squares_SGD(y, tx, gamma, max_iters, batch_size):
    """
	Linear regression using stochastic gradient descent
	@param gamma: learning rate
	@param max_iters: maximum number of iterations
	@param batch_size: the size of batchs used for calculating the stochastic gradient
    @return: the minimum value of mse and the w for which it is attained
    """
    initial_w = np.zeros(tx.shape[1])
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
    return min_loss, w
	

#####################################################
#           		LEAST SQUARES					#
#####################################################

def least_squares(y, tx):
    """
	Least squares regression using normal equations
    @return: minimum mse, and optimal weights
	"""
    w = np.linalg.inv(tx.T.dot(tx)).dot(tx.T).dot(y)
    mse = compute_loss(y,tx,w)
    return mse, w

#####################################################
#           	RIDGE REGRESSION					#
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
	@return:
	"""
    N = y.shape[0]
    w = np.linalg.inv(tx.T.dot(tx) + lambda_*2*N*np.identity(tx.shape[1])).dot(tx.T).dot(y)
    loss = compute_loss_ridge(y,tx,w,lambda_)
    return loss, w
	
	
#####################################################
#           	LOGISTIC REGRESSION					#
#####################################################

def sigmoid(t):
    """apply sigmoid function on t."""
    return np.exp(t)/(1+np.exp(t))

def calculate_log(y, tx, w):
    """compute the cost by negative log likelihood."""
    a = np.dot(tx,w)
    return sum( np.log(1+np.exp(a)) - np.multiply(y,a) )

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

def logistic_regression(y, tx, gamma, max_iter, batch_size):
	"""
	@param gamma: learning rate
	@param max_iter: maximum nuber of iterations
	@param batch_size: the size of batchs used for calculating the stochastic gradient
	@return : the minimum loss and the value w for which it is attained
	"""
    losses = []
    w = np.zeros((tx.shape[1],1))
    y_batch = np.zeros((batch_size,1))
    for iter in range(max_iter):
        batch = batch_iter(y, tx, batch_size, num_batches=1, shuffle=True)
        y_batch[:,0],tx_batch = next(batch)
        loss, w = log_gradient_descent(y_batch, tx_batch, w, gamma)
        losses.append(loss)
        # print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))
    return loss,w
	
#####################################################
#           	LOGISTIC REGRESSION					#
#####################################################

def reg_logistic_regression(y, tx, lambda_, gamma, max_iters):
	raise NotImplementedError