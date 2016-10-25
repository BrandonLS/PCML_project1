import numpy as np
from helpers import batch_iter

def sigmoid(t):
    """apply sigmoid function on t."""
    return np.divide(np.exp(t),1+np.exp(t))

    
def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
#    res2 = 0.0
#    for n in range(y.shape[0]):
#        res2 += np.asscalar(np.log1p(np.exp(tx[n,:].dot(w)))-y[n]*tx[n,:].dot(w))
#        res2 += np.asscalar(np.log1p(np.exp(tx[n,:].dot(w))))

#    return np.asscalar(res)
    # The data set is to large to do in one go
#    res = 0
#    split = 50
#    split_ys = np.split(y, split)
#    split_txs = np.split(tx, split)
#    for split_y, split_tx in zip(split_ys, split_txs):
#        res += np.asscalar(np.sum( np.log1p(np.exp(split_tx.dot(w))) - np.multiply(split_y,split_tx.dot(w))))
#    
#    print(res)
#    print(res2)
#    assert int(res2) == int(res)    
#    return res
    
#    res3= np.sum( np.log1p(np.exp(tx.dot(w))) - np.multiply(y,tx.dot(w)))
    a = np.dot(tx,w)
    res3 = np.sum( np.log(1+np.exp(a)) - np.multiply(y,a))

#    print(res2)
#    print(res3/10000)
#    assert int(res2)==int(res3)
#    return res3/10000
    return res3
    
    

    
    
def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    return tx.T.dot(np.subtract(sigmoid(tx.dot(w)),y))
    
def calculate_stochastic_gradient(y, tx, w):
    """compute the gradient of loss."""
    return tx.T.dot(np.subtract(sigmoid(tx.dot(w)),y))
#    res = np.zeros((tx.shape[1],1))
#    N, D = tx.shape
#    for n in range(N):
#        p1 = tx[n,:].T.dot(w)
#        p2 = sigmoid(p1)
#        p3 = p2 - y[n]
#        p4 = tx[n,:]*p3
#        p4 = p4.reshape(tx.shape[1],1)
#        res = np.add(res, p4)
#        res = np.add(res, (tx[n,:]*(sigmoid(tx[n,:].dot(w))-y[n])).reshape(D,1))
#    return res
    
    

def learning_by_gradient_descent(y, tx, w, alpha):
    """
    Do one step of gradient descen using logistic regression.
    Return the loss and the updated w.
    """
    loss = calculate_loss(y, tx, w)
    gradient = calculate_gradient(y, tx, w)
    w = w - alpha * gradient
    return loss, w
    
def learning_by_stochastic_gradient_descent(y, tx, w, alpha):
    """
    Do one step of gradient descen using logistic regression.
    Return the loss and the updated w.
    """
    loss = calculate_loss(y, tx, w)
    gradient = calculate_stochastic_gradient(y, tx, w)
    w = w - alpha * gradient
    return loss, w
    
def logistic_regression_gradient_descent(y, tx, max_iter=10000, threshold=1e-8, alpha=0.001):
    losses = []

    # build tx
    # tx = np.c_[np.ones((y.shape[0], 1)), x]
    w = np.zeros((tx.shape[1], 1))

    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w, alpha)
        # log info
        if iter % 1000 == 0:
            print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))
        # converge criteria
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    # visualization
    # visualization(y, x, mean_x, std_x, w, "classification_by_logistic_regression_gradient_descent")
    l=calculate_loss(y, tx, w)
    print("The loss={l}".format(l))
    return l, w
    
def logistic_regression_stochastic_gradient_descent(y, tx, initial_w, max_iter=10000, threshold=1e-8, alpha=0.001, batch_size=250):
    losses = []

    w = initial_w
    
    min_loss = 1000000
    min_w = initial_w
    
    # start the logistic regression
    count = 0
    while count < max_iter:
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            # get loss and update w.
#            loss = calculate_loss(y, tx, w)
#            gradient = calculate_stochastic_gradient(minibatch_y, minibatch_tx, w)
#            w = np.subtract(w, alpha * gradient)
            loss, w = learning_by_stochastic_gradient_descent(minibatch_y, minibatch_tx, w, alpha)
            # log info
            if count % 1 == 0:
                print("Current iteration={i}, the loss={l}".format(i=count, l=loss))
            # converge criteria
            losses.append(loss)
            if loss < min_loss:
                min_loss = loss
                min_w = w
            count += 1
            if count > max_iter:
                break
            if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
                break
            
#    l=calculate_loss(y, tx, w)
#    print("The loss={l}".format(l))
    
    return min_loss, min_w
    
    
