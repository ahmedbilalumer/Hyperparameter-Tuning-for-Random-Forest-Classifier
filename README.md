# Hyperparameter Tuning for Random Forest Classifier

This project demonstrates hyperparameter tuning for a Random Forest Classifier using Grid Search and Random Search. The dataset used in this project consists of email data with features representing word counts and a target variable indicating a prediction.

## Installation

To run this project, you need Python 3.x installed on your system. Additionally, install the necessary Python packages using the following command:

## bash
pip install pandas scikit-learn

## Usage:
1. Clone the repository: 
git clone https://github.com/ahmedbilalumer/Hyperparameter-Tuning-for-Random-Forest-Classifier.git
cd Hyperparameter-Tuning-for-Random-Forest-Classifier

2. Update the dataset path:
In main.py, update the file_path variable to point to the location of your emails.csv file.

3. Run the script:
python main.py

## Project Structure
.
├── main.py
├── emails.csv
└── README.md

main.py: The main script that performs hyperparameter tuning and evaluation.
emails.csv: The dataset file.
README.md: Project documentation.
requirements.txt: List of required Python packages.

## Dataset
The dataset (emails.csv) consists of email data with features representing word counts and a target variable Prediction. The structure of the dataset is as follows:

Email No.: Identifier for each email (to be dropped before training).
Prediction: Target variable indicating the prediction.

All other columns represent word counts for various words.

## Hyperparameter Tuning
The project uses two methods for hyperparameter tuning:

# 1. Grid Search:

A systematic approach to try different combinations of parameters from a predefined grid.

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 2. Random Search:

A randomized approach to sample different combinations of parameters from a specified distribution.

param_dist = {
    'n_estimators': [int(x) for x in range(50, 201)],
    'max_depth': [None] + [int(x) for x in range(10, 31)],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

## Evaluation
The best model from both Grid Search and Random Search is evaluated on a test set, and the accuracy is reported.

print("Best parameters from Grid Search:", best_params_grid)
print("Accuracy from Grid Search model:", accuracy_grid)
print("Best parameters from Random Search:", best_params_random)
print("Accuracy from Random Search model:", accuracy_random)

