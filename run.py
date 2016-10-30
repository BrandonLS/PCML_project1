import numpy as np
from helpers import *

DATA_TRAIN_PATH = '/home/acy/PCML/projects/project1/train.csv' # TODO: download train data and supply path here 
y, tX, ids = load_csv_data(DATA_TRAIN_PATH) #load train data

DATA_TEST_PATH = '/home/acy/PCML/projects/project1/test.csv' # TODO: download train data and supply path here 
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH) #load test data

OUTPUT_PATH = '/home/acy/PCML/projects/project1/predictions.csv' # TODO: fill in desired name of output file for submission

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=1 up to j=degree."""
    N = x.shape[0]
    D = x.shape[1]
    
    phi = np.zeros((N,D*degree))
    col = 0
    for i in range(1,degree+1):
        for j in range(0,D):
            phi[:,col]=x[:,j]**i
            col=col+1
    return phi

def add_feature(x,f):
    """Append the features given in f to the input x"""
    N = x.shape[0]
    D = x.shape[1]
    F = f.shape[1]
    
    res = np.zeros((N,D+F))
    res[:,0:D] = np.copy(x)
    for i in range(0,F):
        res[:,D+i]=f[:,i]
    return res

def find_outlier(tX):
    """returns the index of outliers (-999) in tX"""
    N = tX.shape[0]
    D = tX.shape[1]
    outlier_index = [[] for i in range(D)]
    for d in range(D):
        for n in range(N):
            if tX[n,d] == -999.0:
                outlier_index[d].append(n)     
    return outlier_index

def clean_data(tX,outlier_index):
    """Replace the outliers with zeros in tX. """
    D = tX.shape[1]
    tX_clean =np.copy(tX)
    for d in range(D):
        for n in outlier_index[d%30]:
            tX_clean[n,d] = 0 
    return tX_clean

def normalize(x,mean_x,std_x):
    """Standardize the original data set."""
    if mean_x is None:
        mean_x = np.mean(x, axis=0)
    x = x - mean_x
    if std_x is None:
        std_x = np.std(x, axis=0)
    x[:, std_x>0] = x[:, std_x>0] / std_x[std_x>0]
    return x

def append_ones(x):
    """Append a column of ones to x"""
    tx = np.hstack((np.ones((x.shape[0],1)), x))
    return tx

def feature_process(tX_train, tX_test, poly_degree):
    """Transforms the train and test features"""
    N_tr = tX_train.shape[0]
    N_te = tX_test.shape[0]
    # merge the train and test data to apply the same transformations on both
    tX_merged = np.append(tX_train,tX_test,axis=0)
    
    print('finding outliers')
    # find the outliers in the data
    outlier_index= find_outlier(tX_merged)
    
    print('building polynomials')
    # do the polynomial transformation according to the degree given in input
    phiX_merged = build_poly(tX_merged, poly_degree)
    # append the square root of the features. take their absolute value first to make sure the results are real numbers 
    phiX_merged = add_feature(phiX_merged,np.sqrt(np.abs(tX_merged)))
    # append log(x). add a small bias number to prevent log 0  
    phiX_merged = add_feature(phiX_merged,np.log(np.abs(tX_merged)+0.01))
    # append 1/x. add a small bias number to prevent 1/0 
    phiX_merged = add_feature(phiX_merged,1/(np.abs(tX_merged)+0.01))
    # append 1/x^2. add a small bias number to prevent 1/0 
    phiX_merged = add_feature(phiX_merged,1/(np.abs(tX_merged)+0.01)**2)
    # append hyporbolic tangent of x.
    phiX_merged = add_feature(phiX_merged,np.tanh(tX_merged))
    # append discrete cosine transform of x
    phiX_merged = add_feature(phiX_merged,np.fft.fft(tX_merged,axis=1).real)

    print('cleaning the outliers')
    # clean the outliers 
    phiX_merged = clean_data(phiX_merged,outlier_index)
    print('normalizing the features')
    # normalize the features 
    phiX_merged = normalize(phiX_merged, None, None)
    print('cleaning the outliers once more')
    # replace outliers with zeros one more time normalization changed their values 
    phiX_merged = clean_data(phiX_merged,outlier_index)
    print('appending')
    # append a column of ones
    phiX_merged = append_ones(phiX_merged)
    
    return phiX_merged[0:N_tr,:], phiX_merged[N_tr:N_tr+N_te,:]

def build_k_indices(N, k_fold, seed):
    """build k indices for k-fold."""
    num_row = N
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def sigmoid(t):
    """apply sigmoid function on t."""
    expo = np.exp(t)
    sigm = expo/(1+expo)
    # in case exp(t) overflows, return 1.0
    if np.inf in expo:
        for i, x in enumerate(expo):
            if x == np.inf:
                sigm[i] = 1.0
    return sigm

def calculate_log(y, tx, w):
    """compute the cost by negative log likelihood."""
    loss = sum( np.log(1+np.exp(np.dot(tx,w))) - np.multiply(y,np.dot(tx,w)) )
    # in case exp(np.dot(tx,w)) overflows, return its approximation
    if loss==np.inf:
        loss = sum( np.dot(tx,w) - np.multiply(y,np.dot(tx,w)) )
        print("LOSS OVERFLOW!!")
    return loss

def calculate_log_gradient(y, tx, w):
    """compute the gradient of loss."""
    return np.dot(tx.T,sigmoid(np.dot(tx,w))-y)

def mult_diag(A,d):
    """multiplies a NxN matrix A with a diagonal matrix d in the form of Nx1, where each diagonal elements are stored"""
    N=A.shape[0]
    D=A.shape[1]
    res = np.empty(A.shape)
    for i in range(N):
        res[i,:] = np.multiply(A[i,:],d)
    return res

def calculate_hessian(y, tx, w):
    """return the hessian of the loss function."""
    N = tx.shape[0]
    D = tx.shape[1]
    S = np.zeros([N])
    for n in range(N):
        sigm = sigmoid(np.dot(tx[n,:],w))
        S[n] = sigm*(1-sigm)
    return np.dot(mult_diag(tx.T,S),tx)

def log_gradient_descent_newton(y, tx, w, gamma, lambda_):
    """update the weights with Newton's method based on gradient and hessian calculated"""
    loss = calculate_log(y, tx, w)
    grad = calculate_log_gradient(y, tx, w)
    hessian = calculate_hessian(y, tx, w)
    w = w - gamma * np.dot(np.linalg.inv( hessian+lambda_*np.eye(hessian.shape[0]) ), grad) 
    return loss,w

def logistic_regression_newton(y, tx, gamma, lambda_, max_iter):
    """compute the weights with Newton's method"""
    w = np.zeros((tx.shape[1],1))
    for iter in range(max_iter):
        loss,w = log_gradient_descent_newton(y, tx, w, gamma, lambda_)
        print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))
    return loss,w


### Set the hyper-parameters ###
# polynomial degree
poly_degree = 5
# Regularization factor for logistic regression
lambda_ = 7.742e-06
# step-size of gradient descent
gamma = 0.1
# maximum number of iterations for gradient descent
max_iters = 100
# number of fold for cross-validation
k_fold = 4

# Do the feature processing
tX_train,tX_test_proc = feature_process(tX, tX_test, poly_degree)

# change the labels from (-1,1) to (0,1)
y_train = np.array([1 if i==1 else 0 for i in y])

# number of training samples
N = tX_train.shape[0]
print(N)
# number of features
D = tX_train.shape[1]
print(D)

#seed value for random indice generator
seed = 1
#randomly select indices for the k-fold cross-validation
k_indices = build_k_indices(N, k_fold, seed)

# number of training samples for each fold
N_fold = N * (k_fold-1) / k_fold
# number of test samples for each fold
N_test = N / k_fold

#holds the accuracies obtained in each fold
acc = []
#holds the weights obtained in each fold
ws=np.zeros( (D, k_fold) )
for k in range(k_fold):
    #initialize the training y,tX for the fold
    yTr=np.zeros( (N_fold,1) )
    xTr=np.zeros( (N_fold,D) )
    st_ind=0
    for i in range(k_fold):
        if i==k:
            #choose test y and tX for this fold
            yTe = y_train[k_indices[i]]
            xTe = tX_train[k_indices[i]]
        else:
            #choose train y and tX for this fold
            yTr[st_ind:st_ind+N/k_fold,0] = y_train[k_indices[i]]
            xTr[st_ind:st_ind+N/k_fold] = tX_train[k_indices[i]]
            st_ind = st_ind + N/k_fold

    # calculate the weights and loss for this fold with Newton's method
    loss, w = logistic_regression_newton(yTr, xTr, gamma, lambda_, max_iters)
    # compute the regression with weights calculated
    y_est = sigmoid(np.dot(xTe,w)) 
    # label the estimations as 0 or 1
    y_label = [0 if i<0.5 else 1 for i in y_est]
    ws[:,k:k+1] = w
    # compare if the estimated labels are correct or not
    corr = [True if i==yTe[ind] else False for ind, i in enumerate(y_label)]
    # calculate the accuracy of this fold.
    acc.append(sum(corr)/N_test)
    print("Fold: {f}, Accuracy: {acc}, Loss:{loss}".format(f=k, acc=acc[k], loss=loss))

# calculate the average accuracy of all folds
acc_avg = sum(acc)/k_fold
print("Gamma: {gamma}, Average Accuracy: {acc}".format(gamma=gamma, acc=acc_avg))    

# calculate the final weights by averaging the weights of each fold
weights=np.zeros([D,1])
for d in range(D):
    weights[d,0] = sum(ws[d,:])/k_fold
    
# compute the regression with average weights
y_est = sigmoid(np.dot(tX_test_proc,weights))
# label the predictions as 1 or 1
y_pred = np.array([-1 if i<0.5 else 1 for i in y_est])
# write the predictions to the csv file to submit
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)