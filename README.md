# ML_Project_1
Project 1 of the Machine Learning course given at the EPFL Fall 2020. 

## Higgs Boson: Machine Learning Challenge

### Structure of repository: 
- **data/**: 
	- `test.csv` the test data and `train.csv` the training data
	- **results/**: 
		- **predictions/**: predicted labels of simulations
        - **plots/**: plots with evolution of losses during training/validation		
        - **weights/**: best weights to predict labels
- **source/**: 
	- `implementations.py`: contains implementations of Least Squares Regression (Normal, with GD/SGD), Ridge Regression (Normal) and (Regularized) Logistic Regression with GD
	- `main.py`: main executable
	- **helper_functions/**: 
		- `helper_functions.py`: helper functions needed to run the project
		- `losses.py`: different losses (MSE, Logisitic) and their gradients
		- `ml_methods_labs.py`: methods from the Machine Learning labs
	- **parameters/**: 
		- `default_parameters.json`: JSON with default parameters for each method
		- `parameters.json`: JSON file to input specific parameters

### Instructions to run: 
- Python modules requirements: Numpy, matplotlib.pyplot, datetime, json **TO COMPLETE**
- Decide wether to use default parameters for training or your own. Have a look if the default parameters in `parameters/default_parameters.json` are good for you. Otherwise, to take your own, modify the values in `parameters/parameters.json` to suit you. 
- Go into `source/` and run `$python main.py` andd follow instructions
- Predictions will be saved in `../data/results/predictions/.` and weights will be saved in `../data/results/weights/.`. To see plots of evolution of losses during training, see plots saved in `../data/results/plots/.`
