# Imports
import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import date

# Additional functions
from helper_functions.helper_functions import *
from helper_functions.ml_methods_labs import *
from helper_functions.losses import *
from implementations import *

# Constants
DATA_TRAIN_PATH = 'data/train.csv'
DATA_TEST_PATH = 'data/test.csv'

# List of available methods
models = {
    'LS_GD': 'Least squares with GD',
    'LS_SGD': 'Least squares with SGD',
    'LS_normal': 'Least squares with normal equations',
    'RR_normal': 'Ridge regression with normal equations',
    'LR': '(Regularized) Logistic regression with gradient descent'
}
model = 'LR'

print('<----------------------START OF PROGRAM: :) ----------------------------->')
# Creation of usefull directories
print('Creation of correct directories if not existing')
print('----------------------------------------------')
if not os.path.exists('data/results/predictions'):
    os.makedirs('data/results/predictions')
if not os.path.exists('data/results/plots'):
    os.makedirs('data/results/plots')
if not os.path.exists('data/results/weights'):
    os.makedirs('data/results/weights')
if not os.path.exists('helper_functions'):
    os.makedirs('helper_functions')
    
# Loading of parameters: 
print('Loading parameters')
print('----------------------------------------------')
with open('data/parameters_best_model.json') as json_file:
    parameters = json.load(json_file)
print(f'The program is running {model}: {models[model]}')
print('----------------------------------------------')
#----------------------------------Training preprocessing---------------------
# Data loading
print('Please wait:')
print('Loading data :) this may take a few minutes...')
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

print('----------------------------------------------')
print('Pre-processing data. This may also take a few seconds...')
# Encoding of y to transforms values in the set [0,1]
y_enc = (y+1)/2
y = y_enc

# Pre-processing of training and test data:
poly = parameters[model]['poly']
tX_std, tX_test_std, y = pre_processing(tX_test, tX, y, poly)

#----------------------------------Training-------------------------------
print('----------------------------------------------')
# Model training: 
best_w, avg_loss = train_model(tX=tX_std,
                               y=y,
                               model=model,
                               initial_w=np.zeros(tX.shape[1] * poly + 1),
                               param=parameters)


#----------------------------------Test Preprocessing------------------------------
print('----------------------------------------------')
print('Generating predictions:')

#----------------------------------Predictions: ------------------------------
# Generating the predictions result file
test_prediction = predict_labels(best_w, tX_test_std)

# Generating the name of the file: 
today = date.today().strftime('%m-%d')
params = ''
for param in parameters[model]:
    params += param + '=' + str(parameters[model][param]) + ','
OUTPUT_PATH = 'data/results/predictions/y_pred_' + model + '_' + str(
    today) + '_' + params + '.csv'
np.save('data/results/weights/best_w_'+ model + '_' + str(
    today) + '_' + params + '.npy', best_w)
create_csv_submission(ids_test, test_prediction, OUTPUT_PATH)
print('----------------------------------------------')
print('Predictions saved in ' + 'data/results/')