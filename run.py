"""
ML pipeline
"""
from scripts import logistic_regression, sigmoid
import numpy as np

# TRAINING

# Import data
from helpers import *
DATA_TRAIN_PATH = 'data/train.csv' # TODO: download train data and supply path here 
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

# Clean and standardise data
# We standardise the data matrix, add a column of 1s to the data matrix, and change the -1s to 0s in y
# We then replace all the outliers (-999) with 0
y_shifted, tX_norm = pre_process_data(y, tX)

# Run logisitic regression algorithm
max_iters = 100
batch_size = N/100
gamma=0.00035938136638046257

loss, w = logistic_regression(y_shifted, tX_norm, gamma, max_iters, batch_size)

# PREDICTIONS

#Load test data
DATA_TEST_PATH = 'data/test.csv' # TODO: download train data and supply path here 
OUTPUT_PATH = 'data/output2.csv' # TODO: fill in desired name of output file for submission
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

# Process test data
_, tX_test = pre_process_data(_,tX_test)

# Make predictions
y_est = sigmoid(np.dot(tX_test,w))
y_label = [-1 if i<0.5 else 1 for i in y_est]
create_csv_submission(ids_test, y_label, OUTPUT_PATH)
