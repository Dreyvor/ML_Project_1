# Useful starting lines
import numpy as np
import matplotlib.pyplot as plt
import datetime
# importing the module
import json
import os
from helper_functions.ml_methods_labs import *
from helper_functions.losses import *


def accuracy(features, w, true_y):
    """
    accuracy: calculates the accuracy of a prediction
    @input: 
    - np.array(N,m) features
    - np.array(m,) weights
    - np.array(N,) true_y
    @output: (TP+TN)/Total
    """
    y_pred = predict_labels(w, features)
    #encode to 0/1
    y_pred_enc = (y_pred + 1) / 2
    P_N = len(y_pred_enc[np.where(np.subtract(y_pred_enc, true_y) == 0)])
    return (P_N / len(true_y)) * 100

def build_k_indices(y, k_fold, seed=2):
    """build_k_indices: build k indices for k-fold
    @input: 
    - np.array(N,) y: labels
    - double k_fold: number of k-folds (e.g. 5)
    - double seed: seed for random generator
    @output: k indices sets for k-fold
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation_sets(tX, y, k_indices, i):
    """
    cross_validation_sets: separates tX and y randomly into training and validation sets.
    @input:
    - np.array(N,m) tx: features
    - np.array(N,) y: labels
    - list k_indices: indices for k-fold cross-val
    - i: the index of the iteration. (the i-th iteration to find the correct k_indices)
    @output: 
    - np.array(percentage*N,m) tX_train: training features
    - np.array(percentage*N,) y_train: training labels
    - np.array((1-percentage)*N,m) tX_val: validation features
    - np.array((1-percentage)**N,) y_val: validation labels
    """
    train_indices = np.concatenate(np.delete(k_indices, i, axis=0), axis=0)
    val_indices = k_indices[i]
  
    #creates training and validation:
    tX_train = np.take(tX, train_indices, axis=0)
    y_train = np.take(y, train_indices, axis=0)
    tX_val = np.take(tX, val_indices, axis=0)
    y_val = np.take(y, val_indices, axis=0)

    # tests: 
    size = len(train_indices) + len(val_indices)

    assert (tX_train.shape[0] + tX_val.shape[0] == size)
    assert (y_train.shape[0] + y_val.shape[0] == size)
    
    return tX_train, y_train, tX_val, y_val


def poly_feats(tX, degree): 
    """poly_feats: performs polynomial feature expansion
    @input: 
    - np.array(N,m) tX: features
    - double degree: degree of expansion
    @output: [1, X, X^2, X^3, etc]
    """
    #add bias term:
    if not np.array_equal(tX[:, 0], np.ones(len(tX))):
        tX_poly = np.hstack((np.ones((len(tX), 1)), tX))
    if degree>1: 
        for deg in range(2, degree+1):
              tX_poly = np.c_[tX_poly, np.power(tX, deg)]
    return tX_poly



def replace_invalid_values(tX, invalid_identifier, mean=True):
    """replace_invalid_values: replaces invalid values with the median/mean of all the values in the cooresponding feature
    @input: 
    - np.array(N,m) tX: features
    - double invalid_identifier: invalid data, here -999
    - bool mean: true if replace with the mean, false for median
    @output: np.array(N,m) with invalid data replaced
    """
    means = []
    data = tX.copy()
    new_data = np.empty((tX.shape[0], 1))

    for i in range(0, tX.shape[1]):
        column = data[:, i].copy()
        column = np.delete(column, np.where(column == invalid_identifier))
        if mean:
            data[:, i] = np.where(data[:, i] == invalid_identifier,
                                  np.mean(column), data[:, i])
            new_data = np.concatenate(
                (new_data, np.reshape(data[:, i], (tX.shape[0], 1))), axis=1)
            means.append(np.mean(column))
        else:
            data[:, i] = np.where(data[:, i] == invalid_identifier,
                                  np.median(column), data[:, i])
            new_data = np.concatenate(
                (new_data, np.reshape(data[:, i], (tX.shape[0], 1))), axis=1)
            means.append(np.median(column))
    assert (new_data[:, 1:].shape == tX.shape)
    return new_data[:, 1:], means

def replace_outlayers_values(features, delta, mean=True):
    """replace_outlayers_values: replaces outlier values with the median/mean of 
    all the values in the cooresponding feature
    @input: 
    - np.array(N,m) features: features
    - double delta: a bigger delta increase the tolerance
    - bool mean: true if replace with the mean, false for median
    @output: np.array(N,m) with outlier data replaced
    """
    vals = []
    new_data = np.empty((len(features), 1))
      
    for i in range(features.shape[1]):
        column = features[:, i].copy()
        Q1 = np.quantile(column, 0.10)
        Q3 = np.quantile(column, 0.75)
        IQR = Q3 - Q1
        if mean: new_val = np.mean(column)
        else: new_val = np.median(column)
        col_no_outl = np.where(
            ~((column < (Q1 - delta * IQR)) | (column >
                               (Q3 + delta * IQR))),
            column, new_val)
        new_data = np.concatenate(
            (new_data, np.reshape(col_no_outl, (len(features), 1))),
            axis=1)
        vals.append(new_val)
    return new_data[:, 1:], vals

def standardize(x):
    """standardize data
    @input: 
    - np.array(N,m) x: features
    @output: np.array(N,m) with standardized data
    """
    # if bias column: 
    if np.array_equal(x[:, 0], np.ones(len(x))):
        centered_data = x[:,1:] - np.mean(x[:,1:], axis=0)
        std_data = centered_data / np.std(centered_data, axis=0)
        return np.hstack((np.ones((len(x),1)),std_data))
    else:
        centered_data = x - np.mean(x, axis=0)
        std_data = centered_data / np.std(centered_data, axis=0)
        return std_data

def standardize_with_mean_std(x, mean, std):
    """standardize data
    @input: 
    - np.array(N,m) x: features
    - np.array(N,m) mean: mean
    - np.array(N,m) std: standard deviation
    @output: np.array(N,m) with standardized data
    """
    # if bias term:
    if np.array_equal(x[:, 0], np.ones(len(x))):
        std_data = (x[:,1:] - mean)/std
        return np.hstack((np.ones((len(x),1)), std_data))
    else:
        std_data = (x - mean)/std
        return std_data