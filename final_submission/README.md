# Machine Learning Project 1 - Higgs Boson
Project 1 of the Machine Learning course given at the EPFL Fall 2020. 

## Team Members
- Marijn VAN DER MEER
- Bradley MATHEZ
- Timothée DURAN

## Idea: 
In this project, we implemented simple machine learning models: Least Squares Regression (Normal, with GD/SGD), Ridge Regression (Normal) and (Regularized) Logistic Regression with GD (c.f. `implementations.py`) . Furthermore, we optimized one of them, Logistic Regression to do binary classification on the Higgs Boson experiment challenge data-set (c.f. `run.py`). For this, before optimizing the model, we did some data pre-processing and feature engineering. Those steps can be seen in the helper function `pre_processing` in `helper_functions/helper_functions.py` and in our report. 

## Structure of the repository: 
- `implementations.py`: contains implementations of Least Squares Regression (Normal, with GD/SGD), Ridge Regression (Normal) and (Regularized) Logistic Regression with GD
- `run.py`: main executable to recreate our best score on AICrowd
- `main.py`: user friendly executable to run other models than the one that gave the best performance
- `project_1.ipynb`: notebook with plots for the report
- `README.md`
- **helper_functions/**: 
	- `helper_functions.py`: helper functions needed to run the project
	- `losses.py`: different losses (MSE, Logisitic) and their gradients
	- `ml_methods_labs.py`: methods from the Machine Learning labs
- **data/**:
	- `train.csv`: training data
	- `test.csv`: test data
	- `parameters.json`: JSON file with hyper-parameters for running different models in `main.py` 
	- `parameters_best_model.json`: JSON fiel with parameters for best score
	- **results/**: Attention; the directories listed below could be inexistant if you have not run the `run.py` script. `run.py` creates them, otherwise you can create them on your own. 
		- **predictions/**: labels predicted after training models
        - **plots/**: plots with evolution of training/validation losses       
		- **weights/**: weights with best loss found during training

## Instructions to run:
Python modules requirements: `numpy`, `matplotlib.pyplot`, `datetime`, `json`, `csv` and `os`.

Predictions will be saved in `data/results/predictions/` and weights will be saved in `data/results/weights/`. To see plots of evolution of losses during training, see plots saved in `data/results/plots/`

There are two methods to run our program.

1. To reproduce our best score with logistic regression that we submitted on [AIcrowd](https://www.aicrowd.com):
```
python run.py
```

2. We did a small user-friendly interface to run our code with the other models. Please execute:
```
python main.py
``` 
