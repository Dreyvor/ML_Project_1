# Useful starting lines
import numpy as np
import matplotlib.pyplot as plt
import datetime
# importing the module
import json
import os
from helper_functions.ml_methods_labs import *

def cost_logistic(tX, y, w, parameters):
	"""
	cost_logistic: calculates the logistic (regularized) loss
	@input:
	- np.array(N,) y: labels
	- np.array(N,m) tX: features
	- np.array(m,) w: weights
	- dict parameters: dictionnary of required parameters
	@output: 
	- double cost: logistic loss
	"""
	# get parameters:
	lambda_ = parameters['lambda_']

	predictions = sigmoid_activation(tX @ w)
	m = y.shape[0]

	#Take the sum of both costs: error when label=1 + error when label=0
	cost = -y * np.log(predictions) - (1 - y) * np.log(1 - predictions)

	#Take the average cost:
	cost = cost.sum() / m

	# regularizer:
	if lambda_:
		cost += (lambda_ / (2 * m)) * w.T @ w
	return cost

def gradient_MSE(tX, y, w):
	"""
	gradient_MSE: calculates the gradient of the MSE function 
	@input:
	- np.array(N,) y: labels
	- np.array(N,m) tX: features
	- np.array(m,) w: weights
	@output: np.array(m,) gradient of MSE
	"""
	return (-1 / len(y)) * tX.T @ (y - tX @ w)

def MSE_loss(tX, y, w):
	"""
	MSE_loss: calculates the MSE loss
	@input:
	- np.array(N,) y: labels
	- np.array(N,m) tX: features
	- np.array(m,) w: weights
	@output: double, MSE loss
	"""
	assert(w.shape[0] == tX.shape[1])
	assert(y.shape[0] == tX.shape[0])
	MSE = np.square(np.subtract(y,tX @ w)).mean()
	return MSE
	
def sigmoid_activation(z):
	"""
	sigmoid_activation: calculates the sigmoid activation of a vector z
	@output: np.array(m,) 
	"""
	return 1.0 / (1.0 + np.exp(-z))