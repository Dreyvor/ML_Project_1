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
DATA_TRAIN_PATH = 'data/train.csv'
DATA_TEST_PATH = 'data/test.csv'


models = {
    'LS_GD': 'Least squares with GD',
    'LS_SGD': 'Least squares with SGD',
    'LS_normal': 'Least squares with normal equations',
    'RR_normal': 'Ridge regression with normal equations',
    'LR': 'Logistic regression with GD'
}

# Creation of usefull directories
print('Creation of output directories...')
if not os.path.exists('data/results/predictions'):
    os.makedirs('data/results/predictions')
if not os.path.exists('data/results/plots'):
    os.makedirs('data/results/plots')
if not os.path.exists('data/results/weights'):
    os.makedirs('data/results/weights')
if not os.path.exists('helper_functions'):
    os.makedirs('helper_functions')
    


# User interaction:
print('<----------------------START OF PROGRAM: :) ------------------------>')
model = input(
    'Please choose the model you want to run [LS_GD, LS_SGD, LS_normal, RR_normal, LR]. Remember the parameters can be changed in data/parameters.json : \n'
)
print(f'You chose {model}: {models[model]}')
print('----------------------------------------------')
print('Please wait')
print('Loading data :) this may take a few minutes...')

# Load data:
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

# Pre-processing of training data: 

# Encoding of y to transforms values in the set [0,1]
y_enc = (y+1)/2
y = y_enc

# Pre-processing of training data: 
tX_std, tX_test_std, y = pre_processing(tX_test, tX,y) 


# Loading of parameters: 
print('Loading parameters...')
with open('data/parameters.json') as json_file:
    parameters = json.load(json_file)
    
#----------------------------------Training-------------------------------
print('----------------------------------------------')
# Model training: 
poly = parameters[model]['poly']
best_w, avg_loss = train_model(tX=tX_std,
                               y=y,
                               model=model,
                               initial_w=np.zeros(tX.shape[1] * poly + 1),
                               param=parameters)


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
print('Predictions saved in ' + OUTPUT_PATH)