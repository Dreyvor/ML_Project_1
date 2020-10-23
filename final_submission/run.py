# Imports
import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Additional functions
from datetime import date
from helper_functions.helper_functions import *
from helper_functions.ml_methods_labs import *
from helper_functions.losses import *
from implementations import *

# Constants
DATA_TRAIN_PATH = 'source/train.csv'
DATA_TEST_PATH = 'source/test.csv'

# List of available methods
models = {
    'LS_GD': 'Least squares with GD',
    'LS_SGD': 'Least squares with SGD',
    'LS_normal': 'Least squares with normal equations',
    'RR_normal': 'Ridge regression with normal equations',
    'LR': 'Logistic regression with GD'
}

# Creation of usefull directories
print('Creation of output directories...')
if not os.path.exists('results/predictions'):
    os.makedirs('results/predictions')
if not os.path.exists('results/plots'):
    os.makedirs('results/plots')
if not os.path.exists('results/weights'):
    os.makedirs('results/weights')
if not os.path.exists('helper_functions'):
    os.makedirs('helper_functions')
    
# Loading of parameters
print('Loading parameters...')
with open('source/parameters.json') as json_file:
    parameters = json.load(json_file)

# Data loading
print('Loading data...')
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

# Encoding of y to transforms values in the set [0,1]
y_enc = (y+1)/2
y = y_enc

# Model training
print('Training model...')
model = 'LR'
poly = parameters[model]['poly']
best_w, avg_loss = train_model(tX=tX,
                               y=y,
                               model=model,
                               initial_w=np.zeros(tX.shape[1] * poly + 1),
                               param=parameters)
    
invalid_identifier = -999
tX_test_invalid, medians = replace_invalid_values(tX_test,
                                              invalid_identifier,
                                              mean=False)

delta = 1.5
tX_test_filtered, medians = replace_outlayers_values(tX_test_invalid,
                                                     delta,
                                                     mean=False)
tX_test = tX_test_filtered
# Standard par rapport à moyenne et std de train:

poly_X_test =  poly_feats(tX_test, poly)
poly_X_train = poly_feats(tX, poly)

mean_train = np.mean(poly_X_train[:,1:], axis=0)
std_train = np.std(poly_X_train[:,1:] - mean_train, axis=0)

tX_test_std = standardize_with_mean_std(poly_X_test, mean_train, std_train)

# Generating the name of the file
today = date.today().strftime('%m-%d')
params = ''
for param in parameters[model]:
    params += param + '=' + str(parameters[model][param]) + ','

# Generating the predictions result file
test_prediction = predict_labels(best_w, tX_test_std)
OUTPUT_PATH = 'results/predictions/y_pred_' + model + '_' + str(
    today) + '_' + params + '.csv'
create_csv_submission(ids_test, test_prediction, OUTPUT_PATH)
print('Predictions saved in ' + OUTPUT_PATH)