import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
file_path = 'emails.csv'  
data = pd.read_csv(file_path)

# Drop non-feature columns
data = data.drop(['Email No.'], axis=1)

# Define features (X) and target (y)
X = data.drop('Prediction', axis=1)
y = data['Prediction']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = RandomForestClassifier()

# Define the parameter grid for Grid Search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize Grid Search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# Fit Grid Search
grid_search.fit(X_train, y_train)

# Best parameters from Grid Search
best_params_grid = grid_search.best_params_

# Define the parameter distribution for Random Search
param_dist = {
    'n_estimators': [int(x) for x in range(50, 201)],
    'max_depth': [None] + [int(x) for x in range(10, 31)],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize Random Search
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=100, cv=3, n_jobs=-1, verbose=2, random_state=42)

# Fit Random Search
random_search.fit(X_train, y_train)

# Best parameters from Random Search
best_params_random = random_search.best_params_

# Evaluate the best model from Grid Search
best_model_grid = grid_search.best_estimator_
y_pred_grid = best_model_grid.predict(X_test)
accuracy_grid = accuracy_score(y_test, y_pred_grid)

# Evaluate the best model from Random Search
best_model_random = random_search.best_estimator_
y_pred_random = best_model_random.predict(X_test)
accuracy_random = accuracy_score(y_test, y_pred_random)

print("Best parameters from Grid Search:", best_params_grid)
print("Accuracy from Grid Search model:", accuracy_grid)
print("Best parameters from Random Search:", best_params_random)
print("Accuracy from Random Search model:", accuracy_random)
