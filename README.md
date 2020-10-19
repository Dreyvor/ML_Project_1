# ML_Project_1
Project 1 of the Machine Learning course given at the EPFL Fall 2020. 

## Higgs Boson: Machine Learning Challenge

### Structure of repository: 
- **data/**: 
	- *test.csv* the test data and *train.csv* the training data
	- **results/**: 
		- **predictions/**: predicted labels of simulations
		- **weights/**: best weights to predict labels
- **source/**: 
	- *implementations.py*: contains implementations of Least Squares Regression (Normal, with GD/SGD), Ridge Regression (Normal) and (Regularized) Logistic Regression with GD
	- *main.py*: main executable
	- **helper_functions/**: 
		- *helper_functions.py*: helper functions needed to run the project
		- *losses.py*: different losses (MSE, Logisitic) and their gradients
		- *ml_methods_labs.py*: methods from the Machine Learning labs
	- **parameters/**: 
		- *default_parameters.json*: default parameters for each method
		- *parameters.json*: JSON file to input specific parameters

### Instructions to run: 
- go into *source/* and run: *$python main.py*
