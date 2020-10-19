# Imports
import numpy as np
import matplotlib.pyplot as plt
import datetime
import json
import os
#additional functions:
from helper_functions.helper_functions import *
from helper_functions.ml_methods_labs import *
from helper_functions.losses import *
from implementations import *
from datetime import date


models = {
    'LS_GD': 'Least squares with GD',
    'LS_SGD': 'Least squares with SGD',
    'LS_normal': 'Least squares with normal equations',
    'RR_normal': 'Ridge regression with normal equations',
    'LR': 'Logistic regression with GD'
}

if not os.path.exists('../data/results/predictions'):
    os.makedirs('../data/results/predictions')
if not os.path.exists('../data/results/plots'):
    os.makedirs('../data/results/plots')
if not os.path.exists('../data/results/weights'):
    os.makedirs('../data/results/weights')
if not os.path.exists('parameters'):
    os.makedirs('parameters')
if not os.path.exists('helper_functions'):
    os.makedirs('helper_functions')
    

    
# User interaction:
print('<----------------------START OF PROGRAM: :) ------------------------>')
model = input(
    'Please choose the model you want to run [LS_GD, LS_SGD, LS_normal, RR_normal, LR]:\n'
)
print(f'You chose {model}: {models[model]}')
print('----------------------------------------------')
param = input(
    'Do you want to use the default parameters for training or your parameters? Enter yes for default and no for special parameters:\n'
)
assert (param == 'yes' or param == 'no')
if param == 'yes':
    print(f'You chose default parameters')
else:
    print(f'You chose specific parameters')
print('----------------------------------------------')
print('Please wait')
print('Loading data :) this may take a few minutes...')

# Load data:
DATA_TRAIN_PATH = '../data/train.csv'
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
DATA_TEST_PATH = '../data/test.csv'
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

### Encoding of y: 
y_enc = (y+1)/2
y = y_enc

if param == 'yes':
    with open('parameters/default_parameters.json') as json_file:
        parameters = json.load(json_file)
else:
    with open('parameters/parameters.json') as json_file:
        parameters = json.load(json_file)
print('----------------------------------------------')
best_w, avg_loss = train_model(tX=tX,
                               model=model,
                               y=y,
                               initial_w=None,
                               param=parameters)
    
np.save('../data/results/weights/best_w.csv', best_w)

print('Create predictions: saved in ../data/results/')
tX_test_std = test_preprocessing(tX_test, tX, parameters, model)


today = date.today().strftime('%m-%d')
# parameters:
params = ''
for param in parameters[model]:
    params += param + '=' + str(parameters[model][param]) + ','


test_prediction = predict_labels(best_w, tX_test_std)
OUTPUT_PATH = '../data/results/predictions/y_pred_' + model + '_' + str(
    today) + '_' + params + '.csv'
create_csv_submission(ids_test, test_prediction, OUTPUT_PATH)