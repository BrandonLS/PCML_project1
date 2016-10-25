"""
ML pipeline
"""
from scripts import logistic_regression

# Import data
from proj1_helpers import *
DATA_TRAIN_PATH = '/home/acy/PCML/projects/project1/train.csv' # TODO: download train data and supply path here 
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

# Clean data
N = tX.shape[0]
D = tX.shape[1]
no_samples = N

cnt = 0
exclude = []
for n in range(no_samples):
    for d in range(D):
        if tX[n,d] == -999.0:
            exclude.append(n)
            cnt = cnt + 1
            break

y_c=np.delete(y,exclude,0);
tX_c=np.delete(tX,exclude,0); 

# Standardise data
tX_norm, mean, std = standardize(tX_c, None, None)
y_shifted = np.array([1 if i==1 else 0 for i in y_c])

# Run logisitic regression algorithm
max_iters = 100
batch_size = N/100
gamma=0.00035938136638046257

loss, w = logistic_regression(y_c, tX_c, gamma, max_iters, batch_size)

# Output predictions

DATA_TEST_PATH = '' # TODO: download train data and supply path here 
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

OUTPUT_PATH = '' # TODO: fill in desired name of output file for submission
y_pred = predict_labels(weights, tX_test)
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)

