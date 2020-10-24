# Imports
import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Additional functions
from helper_functions.helper_functions import *
from helper_functions.ml_methods_labs import *
from helper_functions.losses import *

# Loading of parameters
with open('data/parameters.json') as json_file:
     default_parameters = json.load(json_file)

# List of available models
models = {
    'LS_GD': 'Least squares with GD',
    'LS_SGD': 'Least squares with SGD',
    'LS_normal': 'Least squares with normal equations',
    'RR_normal': 'Ridge regression with normal equations',
    'LR': 'Logistic regression with GD'
}

#--------------------------------- FUNCTIONS ------------------------------#

def least_squares_GD(y, tX, initial_w, max_iters=None, gamma=None, parameters=None):
    """
    least_squares_GD: training with least squares GD
    @input:
    - np.array(N,) y: labels
    - np.array(N,m) tX: features
    - np.array(m,) initial_w: initial weight vector
    - double max_iters: maximum number of epochs
    - double gamma: step size
    - dict parameters: dictionnary of required parameters
    @output: 
    - np.array(m,) best_w: weights that got the smallest loss during cross-val
    - double avg_loss: average loss over validation sets during cross-val
    """
    model = 'LS_GD'
    if max_iters != None and gamma != None:
        parameters = {
            'LS_GD': {
                "lambda_": 0.00,
                "lr": gamma,
                "max_iters": max_iters,
                "K": 5,
                "poly": 2
            }
        }
    good_shape = 1 + tX.shape[1] * parameters['LS_GD']['poly']
    if initial_w.shape[0] != good_shape:
        print(
            'Wrong shape for initial_w, should be of shape {}. Polynomial expans of deg {} + a bias term'
            .format(good_shape, parameters['LS_GD']['poly']))
        print('We will use initial_w = 0 everywhere instead:')
        poly = parameters['LS_GD']['poly']
        initial_w = np.zeros(tX.shape[1] * poly + 1)
    
    best_w, avg_loss = train_model(tX, y, model, initial_w, parameters)
    return best_w, avg_loss


def least_squares_SGD(y, tX, initial_w, max_iters=None, gamma=None, parameters=None):
    """
    least_squares_SGD: training with least squares SGD
    @input:
    - np.array(N,) y: labels
    - np.array(N,m) tX: features
    - np.array(m,) initial_w: initial weight vector
    - double max_iters: maximum number of epochs
    - double gamma: learning rate
    - dict parameters: dictionnary of required parameters
    @output: 
    - np.array(m,) best_w: weights that got the smallest loss during cross-val
    - double avg_loss: average loss over validation sets during cross-val
    """
    model = 'LS_SGD'
    if max_iters != None and gamma != None:
        parameters = {
            'LS_SGD': {
                "lambda_": 0.00,
                "lr": gamma,
                "max_iters": max_iters,
                "K": 5,
                "batch_size":1,
                "poly": 2
            }
        }
    poly = parameters['LS_SGD']['poly'] 
    good_shape = 1 + tX.shape[1] * poly
    if initial_w.shape[0] != good_shape:
        print(
            'Wrong shape for initial_w, should be of shape {}. Polynomial expans of deg {} + a bias term'
            .format(good_shape, poly))
        print('We will use initial_w = 0 everywhere instead:')
        initial_w = np.zeros(tX.shape[1] * poly + 1)
    best_w, avg_loss = train_model(tX, y, model, initial_w, parameters)
    return best_w, avg_loss


def least_squares(y, tX, parameters=None):
    """
    least_squares: weights with normal equations of least squares
    @input:
    - np.array(N,) y: labels
    - np.array(N,m) tx: features
    @output:
    - np.array(m,) best_w: weights with smallest loss during cross-val 
    - double avg_loss: average loss over validation sets during cross-val
    """
    if parameters == None:
        parameters = {"LS_normal": {"max_iters": 1, "K": 5, "poly": 2}}
    model = 'LS_normal'
    initial_w = np.zeros((1 + tX.shape[1] * parameters['LS_normal']['poly']))
    best_w, avg_loss = train_model(tX=tX,
                                   y=y,
                                   model=model,
                                   initial_w=initial_w,
                                   param=parameters)
    return best_w, avg_loss


def ridge_regression(y, tX, parameters = None):
    """
    ridge_regression: weights with normal equations of ridge regression
    @input:
    - np.array(N,) y: labels
    - np.array(N,m) tx: features
    @output:
    - np.array(m,) best_w: weights with smallest loss during cross-val 
    - double avg_loss: average loss over validation sets during cross-val
    """
    model = 'RR_normal'
    if parameters == None:
        parameters = {"RR_normal": {"max_iters":1,"K":5,"lambda_":0.02, "poly":2} }
    model = 'RR_normal'
    initial_w = np.zeros((1 + tX.shape[1] * parameters['RR_normal']['poly']))
    best_w, avg_loss = train_model(tX=tX,
                                   y=y,
                                   model=model,
                                   initial_w=initial_w,
                                   param=parameters)
    return best_w, avg_loss


def logistic_regression(y, tX, initial_w, max_iters=None, gamma=None, parameters=None):
    """
    logistic_regression: logistic (regularized) regression with GD
    @input:
    - np.array(N,) y: labels
    - np.array(N,m) tX: features
    - np.array(m,) initial_w: initial weight vector
    - dict parameters: dictionnary of required parameters
    @output: 
    - np.array(m,) best_w: weights that got the smallest loss during cross-val
    - double avg_loss: average loss over validation sets during cross-val
    """
    model = 'LR'
    if max_iters != None and gamma != None:
        parameters = {
            "LR": {
                "lambda_": 0.02,
                "lr": 0.18,
                "max_iters": 250,
                "K": 5,
                "poly": 2
            }
        }
    poly = parameters[model]['poly']
    good_shape = 1 + tX.shape[1] * poly
    if initial_w.shape[0] != good_shape:
        print(
            'Wrong shape for initial_w, should be of shape {}. Polynomial expans of deg {} + a bias term'
            .format(good_shape, poly))
        print('We will use initial_w = 0 everywhere instead:')
        initial_w = np.zeros(tX.shape[1] * poly + 1)
    best_w, avg_loss = train_model(tX, y, model, initial_w, parameters)
    return best_w, avg_loss


def reg_logistic_regression(y, tX, lambda_, initial_w, max_iters=None, gamma=None, parameters=None):
    """
    logistic_regression: logistic (regularized) regression with GD
    @input:
    - np.array(N,) y: labels
    - np.array(N,m) tX: features
    - np.array(m,) initial_w: starting weights
    - dict parameters: dictionnary of required parameters
    @output: 
    - np.array(m,) best_w: weights that got the smallest loss during cross-val
    - double avg_loss: average loss over validation sets during cross-val
    """
    model = 'LR'
    if max_iters != None and gamma != None:
        parameters = {
            "LR": {
                "lambda_": 0.02,
                "lr": 0.3,
                "max_iters": 100,
                "K": 5,
                "poly": 2
            }
        }
    poly = parameters[model]['poly']
    good_shape = 1 + tX.shape[1] * poly
    if initial_w.shape[0] != good_shape:
        print(
            'Wrong shape for initial_w, should be of shape {}. Polynomial expans of deg {} + a bias term'
            .format(good_shape, poly))
        print('We will use initial_w = 0 everywhere instead:')
        initial_w = np.zeros(tX.shape[1] * poly + 1)
    best_w, avg_loss = train_model(tX, y, model, initial_w, parameters)
    return best_w, avg_loss
    

#--------------------------------- TRAINING ------------------------------#

def train_model(tX, y, model, initial_w, param=default_parameters, verbose=True, return_acc = False):
    """
    train_model: trains a model according to their parameters in the dictionnary
    @input:
        - np.array(N,m) tX: features
        - np.array(N,) y: labels
        - string model: 'LS_GD', 'LS_SGD', 'LS_normal', 'RR_normal' or 'LR'
        - np.array(m,) initial_w: starting weights for GD and SGD
        - dictionary param: different parameters required for training (lr, lambda, etc)
        - bool verbose: when true prints all data
        - bool return_acc: also returns the average accuracy over k-folds
    @output: 
        - np.array(m,) best_w: weights that got the smallest loss during cross-val
        - double avg_loss: average loss over validation sets during cross-val
    """
    # will keep last weights and last loss
    weights, loss, accs = [], [], []
    parameters = param[model]

    # parameters:
    K, MAX_ITERS, POLY = parameters['K'], parameters['max_iters'], parameters[
        'poly']

    # will keep all losses during training
    costs_ = np.zeros([K, MAX_ITERS])
    train_costs_ = np.zeros([K, MAX_ITERS])

    if verbose:
        print(
            f'{models[model]} starting with: \nInitial_w: {initial_w},\nParameters:{parameters}'
        )
        print('------------------')
        print('START TRAINING:')

    # get indices of k-fold:
    k_indices = build_k_indices(y, K)

    # polynonmial expansion:
    tX_pol = poly_feats(tX, POLY)

    # standardize:
    tX_std = standardize(tX_pol)

    if verbose:
        print(f'Data shape:{tX_std.shape}')

    for i in range(K):
        if verbose:
            print(f'K = {i+1}')

        # create training and validation sets:
        tX_train, y_train, tX_val, y_val = cross_validation_sets(
            tX_std, y, k_indices, i)
        #start with initial_w:
        w = initial_w.copy()
        # keep tabs on losses during training
        cost_history = []
        train_cost_history = []
        accuracy_history = []

        # if RR_normal or LS_normal no GD:
        if model == 'RR_normal':
            w, cost_history = ridge_regression_update(tX_train, y_train,
                                                tX_val, y_val,
                                                parameters)
            cost = cost_history
            accuracy_history = accuracy(tX_val, w, y_val)
            acc = accuracy_history

            if verbose:
                print('Final loss:{:.4f}'.format(cost))
                print('Final accuracy:{:.4f}%'.format(accuracy_history))

        elif model == 'LS_normal':
            w, cost_history = least_squares_update(tX_train, y_train, tX_val,
                                                y_val)
            accuracy_history = accuracy(tX_val, w, y_val)
            cost = cost_history
            acc = accuracy_history
            if verbose:
                print('Final loss:{:.4f}'.format(cost))
                print('Final accuracy:{:.4f}%'.format(accuracy_history))

        # else GD:
        else:
            for j in range(MAX_ITERS):
                if model == 'LR':
                    # calculate loss:
                    cost = cost_logistic(tX_val, y_val, w, parameters)
                    cost_history.append(cost)
                    train_cost_history.append(
                        cost_logistic(tX_train, y_train, w, parameters))
                    # Update weights:
                    w = update_weights_logistic(tX_train, y_train, w,
                                                parameters)
                    acc = accuracy(tX_val, w, y_val)
                    accuracy_history.append(acc)
                if model == 'LS_GD':
                    # calulate loss:
                    cost = MSE_loss(tX_val, y_val, w)
                    cost_history.append(cost)
                    train_cost_history.append(MSE_loss(tX_train, y_train, w))
                    # Update weights:
                    w = update_weights_LS_GD(tX_train, y_train, w, parameters)
                    acc = accuracy(tX_val, w, y_val)
                    accuracy_history.append(acc)
                if model == 'LS_SGD':
                    # calulate loss:
                    cost = MSE_loss(tX_val, y_val, w)
                    cost_history.append(cost)
                    train_cost_history.append(MSE_loss(tX_train, y_train, w))
                    # update weights:
                    w = update_weights_LS_SGD(tX_train, y_train, w, parameters)
                    acc = accuracy(tX_val, w, y_val)
                    accuracy_history.append(acc)
            if verbose:
                print('Final loss:{:.4f}'.format(cost))
                print('Final accuracy:{:.4f}%'.format(acc))
        # Add last weights and loss for auditing:
        weights.append(w)
        accs.append(acc)
        loss.append(cost)
        costs_[i] = cost_history
        if model != 'LS_normal' and model != 'RR_normal':
            train_costs_[i] = train_cost_history
        if verbose:
            print('------------------')

    if verbose:
        print('Average loss: {:.2f}'.format(np.mean(loss)))
        print('Average accuracy: {:.2f}%'.format(np.mean(accuracy_history)))

    # best weights of min loss:
    best_w = weights[np.argmin(loss)]

    # Plot loss evolution for GD:
    if model != 'LS_normal' and model != 'RR_normal' and verbose:
        print('------------------')
        print('Loss evolution:')
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].plot(costs_.T)
        ax[0].set_title(models[model] + ' , validation loss')
        ax[0].legend(range(1, K + 1))
        ax[1].plot(train_costs_.T)
        ax[1].set_title(models[model] + ' , training loss')
        ax[1].legend(range(1, K + 1))
        plt.savefig('data/results/plots/'+model+'.png')
    if return_acc: 
        return best_w, np.mean(loss), np.mean(accs)
    else: 
        return best_w, np.mean(loss)

#--------------------------------- WEIGHTS WITH GD/SGD ------------------------------#

def update_weights_logistic(tX, y, w, parameters):
    """
    update_weights_logistic: one iteration with GD on logistic (regularized) loss
    @input:
    - np.array(N,m) tX: features
    - np.array(N,) y: labels
    - np.array(m,) w: weights to be updated
    - dict parameters: dictionnary of required parameters
    @output: 
    - np.array(m,) w: new weights
    """

    # get parameters:
    lr = parameters['lr']
    lambda_ = parameters['lambda_']
    m = len(tX)

    #Get Predictions:
    predictions = sigmoid_activation(np.dot(tX, w))

    # slope of the cost function across all observations
    gradient = (tX.T @ (predictions - y))

    # if regularization
    if lambda_:
        gradient += lambda_ * w

    gradient /= m

    # Subtract from our weights to minimize cost
    w -= lr * gradient
    return w

def least_squares_update(tX_train, y_train, tX_val, y_val):
    """
    least_squares_update: weights with normal equations of least squares
    @input:
    - np.array(N,m) tX_train and tX_val: training and validation features
    - np.array(N,) y_train and y_val: training and validation labels
    @output: 
    - np.array(m,) w: weights 
    - double loss: MSE loss
    """
    # "train":
    w = np.linalg.solve(tX_train.T @ tX_train, tX_train.T @ y_train)
    # evaluate loss on validation set:
    loss = MSE_loss(tX_val,y_val, w)
    return w, loss

def update_weights_LS_GD(tX, y, w, parameters):
    """
    update_weights_LS_GD: one step of GD with MSE
    @input:
    - np.array(N,m) tX: features
    - np.array(N,) y: labels
    - np.array(m,) w: weights
    - dict parameters: dictionnary of required parameters
    @output: updated weights according to GD
    """
    # get parameters:
    lr = parameters['lr']
    # calculate gradient:
    grad = gradient_MSE(tX, y, w)
    # update weights:
    w -= lr * grad
    return w

def update_weights_LS_SGD(tX, y, w, parameters):
    """
    update_weights_LS_SGD: one step of SGD with MSE
    @input:
    - np.array(N,m) tX: features
    - np.array(N,) y: labels
    - np.array(m,) w: weights
    - dict parameters: dictionnary of required parameters
    @output: updated weights according to SGD
    """   
    # get parameters:
    lr = parameters['lr']
    # batch_size to 1 for SGD, otherwise mini-batch:
    batch_size = parameters['batch_size']

    for minibatch_y, minibatch_tX in batch_iter(y, tX, batch_size):
        # calculate gradient on batch:
        grad = gradient_MSE(minibatch_tX, minibatch_y, w)
        # update weights:
        w -= lr * grad
    return w

def ridge_regression_update(tX_train, y_train, tX_val, y_val, parameters):
    """
    ridge_regression_update: weights with normal equations of least squares
    @input:
    - np.array(N,m) tX_train and tX_val: training and validation features
    - np.array(N,) y_train and y_val: training and validation labels
    - dict parameters: dictionnary of required parameters
    @output:
    - np.array(m,) w: weights 
    - double loss: MSE loss
    """
    # get parameters:
    lambda_ = parameters['lambda_']

    # "train":
    w = np.linalg.solve(
        tX_train.T @ tX_train + lambda_ * np.eye(tX_train.shape[1]),
        tX_train.T @ y_train)

    # evaluate loss on validation set:
    loss = MSE_loss(tX_val, y_val, w)
    return w, loss