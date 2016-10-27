"""
Methods for cross validation
"""
import numpy as np
from implementations import logistic_regression, sigmoid, least_squares_GD, least_squares_SGD, least_squares, ridge_regression, reg_logistic_regression
from helpers import predict_labels
import matplotlib.pyplot as plt

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)




def cross_validation(y, tX, gamma, method='logistic_regression'):
	"""Cross validation for logistic regression
	@param gamma: learning rate
	@return : the average accuracy over the four fold validations
	"""
	N, D = tX.shape
	
	# Logistic regression parameters
	max_iters = 100
	batch_size = N/100

	# Cross validation parameters
	seed = 1
	k_fold = 4
	k_indices = build_k_indices(y, k_fold, seed)

	N_fold = N * (k_fold-1) / k_fold
	N_test = N / k_fold
	
	acc=[]
	
	for k in range(k_fold): 
		yTr=np.array([])
		xTr=np.zeros( (0,D) )
		for i in range(k_fold):
			if i==k:
				yTe = y[k_indices[i]]
				xTe = tX[k_indices[i]]
			else:
				yTr = np.append(yTr, y[k_indices[i]], axis=0)
				xTr = np.append(xTr, tX[k_indices[i]], axis=0)

		initial_w = np.zeros(tX.shape[1])
		if method == 'logistic_regression':
			initial_w = np.zeros((tX.shape[1],1))
			w, loss = logistic_regression(yTr, xTr, initial_w, max_iters, gamma)
			y_est = sigmoid(np.dot(xTe,w))
			y_label = [0 if i<0.5 else 1 for i in y_est]
		elif method == 'reg_logistic_regression':
			initial_w = np.zeros((tX.shape[1],1))
			lambda_ = 0.1
			w, loss = reg_logistic_regression(yTr, xTr, lambda_, initial_w, max_iters, gamma)
			y_est = sigmoid(np.dot(xTe,w))
			y_label = [0 if i<0.5 else 1 for i in y_est]
		elif method == 'least_squares_GD':
			w, loss = least_squares_GD(yTr, xTr, initial_w, max_iters, gamma)
			y_label = predict_labels(w, xTe)
		elif method == 'least_squares_SGD':
			w, loss = least_squares_SGD(yTr, xTr, initial_w, max_iters, gamma)
			y_label = predict_labels(w, xTe)
		elif method == 'least_squares':
			w, loss = least_squares(yTr, xTr)
			y_label = predict_labels(w, xTe)
		elif method == 'ridge_regression':
			w, loss = ridge_regression(yTr, xTr, 0.1)
			y_label = predict_labels(w, xTe)
		else:
			raise Exception('Invalid method')
			
		

		corr = [True if i==yTe[ind] else False for ind, i in enumerate(y_label)]
		acc.append(sum(corr)/N_test)
		# print("Fold: {f}, Accuracy: {acc}, Loss:{loss}".format(f=k, acc=acc[k], loss=loss))
	return (sum(acc)/k_fold), acc

def find_optimal_gamma(y_shifted, tX_norm):
	"""Cross validation for logistic regression"""
	N, D = tX_norm.shape

	max_acc = 0
	opt_gamma = 0
	acc_avg = []
	acc = []
	ind=0
	#gamma_range = np.linspace(0.01, 0.001, 10)
	gamma_range = np.logspace(-3, -5, 10)
	for gamma in gamma_range:
		acc.append([])
		acc_avg_, acc_ = cross_validation(y_shifted, tX_norm, gamma)
		acc_avg.append(acc_avg_)
		acc[ind] = acc_
		print("Gamma: {gamma}, Average Accuracy: {acc}".format(gamma=gamma, acc=acc_avg[ind]))    
		if acc_avg[ind]>max_acc:
			max_acc = acc_avg[ind]
			opt_gamma = gamma
		ind=ind+1
	print("Maximum Accuracy: {max_acc}, Optimal Gamma: {opt_gamma}".format(max_acc=max_acc, opt_gamma=opt_gamma))         
	plt.boxplot(acc,labels = gamma_range); 
	return opt_gamma