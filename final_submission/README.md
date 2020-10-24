# Machine Learning Project 1 - Higgs Boson
Project 1 of the Machine Learning course given at the EPFL Fall 2020. 

## Team Memberes
- Marijn VAN DER MEER
- Bradley MATHEZ
- Timothée DURAN

### Structure of repository: 
- `implementations.py`: contains implementations of Least Squares Regression (Normal, with GD/SGD), Ridge Regression (Normal) and (Regularized) Logistic Regression with GD
- `run.py`: main executable
- **helper_functions/**: 
	- `helper_functions.py`: helper functions needed to run the project
	- `losses.py`: different losses (MSE, Logisitic) and their gradients
	- `ml_methods_labs.py`: methods from the Machine Learning labs
- **data/**:
	- `train.csv` the training data
	- `test.csv` the test data
	- `parameters.json`: JSON file to input specific parameters
	- **results/**: The directories listed below could be inexistant if you still have not run the `run.py` script.
		- **predictions/**: predicted labels of simulations
        - **plots/**: plots with evolution of losses during training/validation
        - **weights/**: best weights to predict labels

### Instructions to run:
Python modules requirements: `numpy`, `matplotlib.pyplot`, `datetime`, `json`, `csv` and `os`.

Predictions will be saved in `data/results/predictions/` and weights will be saved in `data/results/weights/`. To see plots of evolution of losses during training, see plots saved in `data/results/plots/`

There are two methods to run our program.
1) For the project **submission** use:
```
python run.py
```
It will compute our best score that we submitted on (AIcrowd)[https://www.aicrowd.com]
2) We did a small user-friendly interface to run our code with the other models. Please execute:
```
python main.py
``` 
