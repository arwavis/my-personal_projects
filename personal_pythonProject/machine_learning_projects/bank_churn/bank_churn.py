"""
This script performs machine learning classification using various algorithms such as Logistic Regression, K-Nearest Neighbors (KNN),
Decision Tree, Random Forest, Ada Boost, Gradient Boost, XG Boost, and Cat Boost. The dataset is read from 'train.csv', and the model
is trained and evaluated on the data. The code includes data preprocessing, model training, hyperparameter tuning, and prediction on
new data.

Functions:
- log_reg(): Predicts the outcome using Logistic Regression on new data and prints the result.
- knn_reg(): Predicts the outcome using K-Nearest Neighbors on new data and prints the result.
- dt_reg(): Predicts the outcome using Decision Tree on new data and prints the result.
- rf_reg(): Predicts the outcome using Random Forest on new data and prints the result.
- adb_boosting(): Predicts the outcome using Ada Boost on new data and prints the result.
- gb_boosting(): Predicts the outcome using Gradient Boost on new data and prints the result.
- xg_boost(): Predicts the outcome using XG Boost on new data and prints the result.
- cat_boost(): Predicts the outcome using Cat Boost on new data and prints the result.

Note: Some parts of the code are commented out, and certain libraries and plotting functions are not used. Uncommenting and
enabling these sections can provide additional data analysis and visualization.

Author: Aravind Viswanathan
Date: 24/01/2024
"""

# Importing all required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
# from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import warnings

warnings.filterwarnings("ignore")
# import matplotlib.pyplot as plt
# import seaborn as sns
# import math
# import xgboost as xgb
# import shap
# import optuna
# from optuna import Trial
from catboost import CatBoostClassifier
# from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score

# Read Data
data = pd.read_csv("train.csv")
# data.head(2)
test_data = pd.read_csv("test.csv")
# test_data.head(2)

# Data Analysis code - Enable as required.
# data.shape
# data.describe()
# data.info()
# data.isnull().sum()
# data.head()
# data['Surname'].value_counts()
# data['Geography'].value_counts()
# data['Gender'].value_counts()
# Dropping the columns that I feel are not required
# id
# CustomerId
# Surname
# data.head(2)


# Some Plotting exmaples of code.
# DATA Plotting

# plt.pie(data['Exited'].value_counts().values, labels=np.unique(data['Exited']), autopct='%1.1f%%', startangle=90)
# plt.title('Distribution of Exited')
# plt.show()

# sns.scatterplot(x='Gender',y='EstimatedSalary',data=data,hue='Exited')

# Based on description of dataset, can define the list of categorical variables (include binaries)
# categorical_variables = ['id', 'CustomerId', 'Surname', "CreditScore",
#                          'Geography', "Gender", 'Age',
#                          'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
#                          'IsActiveMember', 'EstimatedSalary'
#                         ]

# # Get remaining numeric variables but remove target
# numeric_variables = [col for col in data.columns if col not in categorical_variables]
# numeric_variables.remove('Exited')

# variables = numeric_variables + categorical_variables

# def plot_distributions(data, columns, target_variable):

#     # Determine the number of subplots required
#     num_columns = len(variables)
#     num_subplots = math.ceil(num_columns / 2)

#     # Create subplots
#     fig, axes = plt.subplots(num_subplots, 2, figsize=(12, 4 * num_subplots))
#     axes = axes.flatten()

#     # Plot distribution plots for each numeric column
#     for i in range(num_columns):
#         sns.histplot(data=data, x=columns[i], hue=target_variable, stat='percent', common_norm=False, kde=True, ax=axes[i])
#         axes[i].set_title(f'Distribution of {columns[i]}')

#     # Adjust layout
#     plt.tight_layout()
#     plt.show()

# plot_distributions(data, variables, 'Exited')

# Dropping some columns as they are not required for the model
data = data.drop(["id", "CustomerId", "Surname"], axis=1)
# Converting categorical variables of the dataset into numerical variables - using ONE HOT ENCODING technique
data['Geography'] = data['Geography'].apply({'France': 0, 'Spain': 1, 'Germany': 2}.get)
data['Gender'] = data['Gender'].apply({'Male': 0, 'Female': 1}.get)
# data.corr()
# plt.figure(figsize=(12,8))
# sns.heatmap(data.corr(),annot=True)

# Dividing the dataset into dependent and independent columns
X = data.drop('Exited', axis=1)
y = data['Exited']

## Splitting the dataset into training and testing set
### 20% of the dataset will be used for testing(evaluation) and 80% of the data will be used for training purposes

# Import necessary libraries
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)


# Define a function to scale the data
def scale_data(data):
    """Function to scale data using StandardScaler"""
    # Initialize StandardScaler
    scaler = StandardScaler()
    # Fit and transform data
    return scaler.fit_transform(data)


# Scale the training and testing sets
scaled_X_train = scale_data(X_train)
scaled_X_test = scale_data(X_test)

######################################## Logistic Regression ####################################################
# from sklearn.linear_model import LogisticRegressionCV
# from sklearn.metrics import confusion_matrix, accuracy_score
# import numpy as np

# Initialize the logistic regression model with cross-validation
log_model = LogisticRegressionCV()

# Fit the model to the training data
log_model.fit(scaled_X_train, y_train)

# Make predictions on the test data
log_pred = log_model.predict(scaled_X_test)

# Calculate the accuracy and confusion matrix of the predictions
log_ac = accuracy_score(y_test, log_pred)
log_cm = confusion_matrix(y_test, log_pred)

# Print the accuracy and confusion matrix with proper formatting
rounded_log_ac = np.round(float(log_ac), 2)
print(f"Logistic Regression Accuracy: {rounded_log_ac * 100}%")
print("\nLogistic Regression Confusion Matrix:")
print(log_cm)

############################################## KNN #############################################################
# import numpy as np
# from sklearn.pipeline import Pipeline
# from sklearn._selection import GridSearchCV
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import confusion_matrix, accuracy_score

# Define the pipeline operations
scaler = StandardScaler()
knn = KNeighborsClassifier()
operations = [('scaler', scaler), ('knn', knn)]

# Define the parameter grid for GridSearchCV
k_values = list(range(1, 10))
knn_param_grid = {'knn__n_neighbors': k_values}

# Define the full cross-validation classifier
full_cv_classifier = GridSearchCV(
    Pipeline(operations),  # Use the pipeline
    knn_param_grid,  # Use the parameter grid
    cv=5,  # Use 5-fold cross-validation
    scoring='accuracy'  # Use accuracy scoring
)

# Fit the classifier to the training data
full_cv_classifier.fit(scaled_X_train, y_train)

# Get the best estimator
best_estimator = full_cv_classifier.best_estimator_

# Print the best parameters
print("Best parameters:", best_estimator.get_params())

# Make predictions on the test data
knn_pred = best_estimator.predict(scaled_X_test)

# Calculate the confusion matrix and accuracy score
knn_cm = confusion_matrix(y_test, knn_pred)
knn_ac = accuracy_score(y_test, knn_pred)

# Round the accuracy score to two decimal places
rounded_knn_ac = np.round(float(knn_ac), 2)

# Print the accuracy score and confusion matrix
print(f"KNN Accuracy: {rounded_knn_ac * 100}%")
print("\nKNN Confusion Matrix:")
print(knn_cm)

######################################## Support Vector Machine ####################################################
# I am Disabling this code as it take longer time to run
# svc = SVC(class_weight='balanced')
# svc_param_grid = {'C': [0.001, 0.01, 0.1, 0.5, 1], 'gamma': ['scale', 'auto']}
# svc_grid = GridSearchCV(svc, svc_param_grid)
# svc_grid.fit(scaled_X_train, y_train)
# svc_grid_pred = svc_grid.predict(scaled_X_test)
# svc_cm = confusion_matrix(y_test, svc_grid_pred)
# svc_ac = accuracy_score(y_test, svc_grid_pred)
# rounded_svc_ac = np.round(float(svc_ac), 2)
# print(f"Support Vector Machine Accuracy: {rounded_svc_ac * 100}")
# print("Support Vector Machine Confusion Matrix: ")
# print(svc_cm)

######################################## Decision Tree #######################################################
# from sklearn.metrics import confusion_matrix, accuracy_score
# import numpy as np

# Initialize the decision tree classifier
decision_tree_model = DecisionTreeClassifier()

# Fit the model to the training data
decision_tree_model.fit(scaled_X_train, y_train)

# Make predictions on the test data
decision_tree_pred = decision_tree_model.predict(scaled_X_test)

# Calculate the accuracy and confusion matrix
dt_cm = confusion_matrix(y_test, decision_tree_pred)
dt_ac = accuracy_score(y_test, decision_tree_pred)

# Round the accuracy score to two decimal places
rounded_dt_ac = np.round(dt_ac, 2)

# Print the accuracy score and confusion matrix
print(f"Decision Tree Accuracy: {rounded_dt_ac * 100}%")
print("\nDecision Tree Confusion Matrix:")
print(dt_cm)

######################################## Random Forest Machine ##################################################
# import numpy as np
# from sklearn.metrics import confusion_matrix, accuracy_score
# from sklearn.ensemble import RandomForestClassifier

# Set the number of trees and random state
NUM_TREES = 10
RANDOM_STATE = 101

# Initialize the Random Forest classifier
Random_Forest_model = RandomForestClassifier(n_estimators=NUM_TREES, random_state=RANDOM_STATE)

# Train the model on the scaled training data
Random_Forest_model.fit(scaled_X_train, y_train)

# Make predictions on the test data
Random_Forest_preds = Random_Forest_model.predict(scaled_X_test)

# Calculate the confusion matrix and accuracy score
rf_cm = confusion_matrix(y_test, Random_Forest_preds)
rf_ac = accuracy_score(y_test, Random_Forest_preds)

# Round the accuracy score to two decimal places
rounded_rf_ac = np.round(rf_ac, 2)

# Print the accuracy score and confusion matrix
print(f"Random Forest Machine Accuracy: {rounded_rf_ac * 100}%")
print("\nRandom Forest Machine Confusion Matrix:")
print(rf_cm)

######################################## Boosting - ADA Boost ##################################################
# Import necessary libraries
# from sklearn.metrics import confusion_matrix, accuracy_score
# import numpy as np
# Initialize AdaBoostClass
ada_boost_model = AdaBoostClassifier(n_estimators=15)

# Fit the model with training data
ada_boost_model.fit(scaled_X_train, y_train)

# Make predictions on test data
ada_boost_preds = ada_boost_model.predict(scaled_X_test)

# Calculate evaluation metrics
adb_cm = confusion_matrix(y_test, ada_boost_preds)
adb_ac = accuracy_score(y_test, ada_boost_preds)

# Round the accuracy score to 2 decimal places
rounded_adb_ac = np.round(adb_ac, 2)

# Print the evaluation metrics
print(f"ADA Boost Accuracy: {rounded_adb_ac * 100}%")
print("\nADA Boost Confusion Matrix:")
print(adb_cm)

######################################## Boosting - Gradien Boost ##################################################
# Initialize AdaBoostClass
gb_model = GradientBoostingClassifier()

# Define hyperparameter grid
gb_param_grid = {"n_estimators": [1, 5, 10], 'max_depth': [3, 4, 5, 6]}

# Use GridSearchCV for hyperparameter tuning
gb_grid = GridSearchCV(gb_model, gb_param_grid)

# Fit the model with training data
gb_grid.fit(scaled_X_train, y_train)

# Make predictions using the  test data
gb_predictions = gb_grid.predict(scaled_X_test)

# Calculate the confusion matrix and accuracy score
gb_cm = confusion_matrix(y_test, gb_predictions)
gb_ac = accuracy_score(y_test, gb_predictions)

# Round the accuracy score to two decimal places
rounded_gb_ac = np.round(float(gb_ac), 2)

# Print the evaluation metrics
print(f"Gradien Boost Accuracy: {rounded_gb_ac * 100}")
print("\nGradien Boost Confusion Matrix: ")
print(gb_cm)

############################################# XG Boost ########################################################

# Create XGBClassifier model
xgb_model = XGBClassifier()

# Define hyperparameter grid
xgb_param_grid = {'n_estimators': [1, 5, 10, 20], 'max_depth': [3, 4, 5, 6], }

# Use GridSearchCV for hyperparameter tuning
xgb_grid = GridSearchCV(xgb_model, xgb_param_grid)

# Fit the model with training data
xgb_grid.fit(scaled_X_train, y_train)

# Make predictions using the  model
xgb_predictions = xgb_grid.predict(scaled_X_test)

# Calculate accuracy and confusion matrix
xgb_ac = accuracy_score(y_test, xgb_predictions)
rounded_xgb_ac = np.round(float(xgb_ac), 2)
print(f"XG Boost Accuracy: {rounded_xgb_ac * 100}")

xgb_cm = confusion_matrix(y_test, xgb_predictions)
print("\nXG Boost Confusion Matrix:")
print(xgb_cm)

############################################# CAT Boost #############################################

# Create CatBoostClassifier model
catboost_model = CatBoostClassifier()

# Define hyperparameter grid
catboost_param_grid = {'n_estimators': [1, 5, 10, 20], 'depth': [3, 4, 5, 6]}

# Use GridSearchCV for hyperparameter tuning
catboost_grid = GridSearchCV(catboost_model, catboost_param_grid)
catboost_grid.fit(scaled_X_train, y_train)

# Make predictions using the best model
catboost_predictions = catboost_grid.predict(scaled_X_test)

# Calculate accuracy and confusion matrix
catboost_ac = accuracy_score(y_test, catboost_predictions)
rounded_catboost_ac = np.round(float(catboost_ac), 2)
print(f"CAT Boost Accuracy: {rounded_catboost_ac * 100}")

catboost_cm = confusion_matrix(y_test, catboost_predictions)
print("CAT Boost Confusion Matrix:")
print(catboost_cm)

# Predicting on New Data

data_new = {'CreditScore': 668, 'Geography': 0, 'Gender': 0, 'Age': 33, 'Tenure': 3,
            'Balance': 0.0, 'NumOfProducts': 2, "HasCrCard": 1.0, "IsActiveMember": 0.0, "EstimatedSalary": 181449.97}
index = [1]  # serial number
my_data = pd.DataFrame(data_new, index)


# Function to predict on New Data

def log_reg():
    """Predicts the outcome using Logistic Regression on new data and prints the result."""
    data_output = log_model.predict(my_data.values)

    if data_output == 0:
        print("Predicted using Logistic Regression: The customer will continue ")
    else:
        print("Predicted using Logistic Regression: The customer will exit ")

    print(f"Above data is calculated with an Accuracy {rounded_log_ac * 100}% ")


def knn_reg():
    """Predicts the outcome using K-Nearest Neighbors on new data and prints the result."""
    data_output = full_cv_classifier.predict(my_data.values)

    if data_output == 0:
        print("Predicted using KNN : The customer will continue ")
    else:
        print("Predicted using KNN : The customer will exit ")

    print(f"Above data is calculated with an Accuracy {rounded_knn_ac * 100}% ")


# def svc_reg():
#     data_output = svc_grid.predict(my_data.values)

#     if data_output == 0:
#         if data_output == 0:
#   print("Predicted using Support Vector Machine: The customer will continue ")
# else:
#   print("Predicted using Support Vector Machine: The customer will exit ")

#     print(f"Above data is calculated with an Accuracy {rounded_svc_ac * 100}% ")

def dt_reg():
    """Predicts the outcome using Decision Tree on new data and prints the result."""
    data_output = decision_tree_model.predict(my_data.values)

    if data_output == 0:
        print("Predicted using Decision Tree: The customer will continue ")
    else:
        print("Predicted using Decision Tree: The customer will exit ")

    print(f"Above data is calculated with an Accuracy {rounded_dt_ac * 100}% ")


def rf_reg():
    """Predicts the outcome using Random Forest on new data and prints the result."""
    data_output = Random_Forest_model.predict(my_data.values)

    if data_output == 0:
        print("Predicted using Random Forest: The customer will continue ")
    else:
        print("Predicted using Random Forest: The customer will exit ")

    print(f"Above data is calculated with an Accuracy {rounded_rf_ac * 100}% ")


def adb_boosting():
    """Predicts the outcome using Ada Boost on new data and prints the result."""
    data_output = ada_boost_model.predict(my_data.values)

    if data_output == 0:
        print("Predicted using Ada Boost: The customer will continue ")
    else:
        print("Predicted using Ada Boost: The customer will exit ")

    print(f"Above data is calculated with an Accuracy {rounded_adb_ac * 100}% ")


def gb_boosting():
    """Predicts the outcome using Gradient Boost on new data and prints the result."""
    data_output = gb_grid.predict(my_data.values)

    if data_output == 0:
        print("Predicted using Gradient Boost: The customer will continue ")
    else:
        print("Predicted using Gradient Boost: The customer will exit ")

    print(f"Above data is calculated with an Accuracy {rounded_gb_ac * 100}% ")


def xg_boost():
    """Predicts the outcome using XG Boost on new data and prints the result."""
    data_output = xgb_grid.predict(my_data.values)

    if data_output == 0:
        print("Predicted using XG Boost: The customer will continue ")
    else:
        print("Predicted using XG Boost: The customer will exit ")

    print(f"Above data is calculated with an Accuracy {rounded_xgb_ac * 100}% ")


def cat_boost():
    """Predicts the outcome using Cat Boost on new data and prints the result."""
    data_output = catboost_grid.predict(my_data.values)

    if data_output == 0:
        print("Predicted using Cat Boost: The customer will continue ")
    else:
        print("Predicted using Cat Boost: The customer will exit ")

    print(f"Above data is calculated with an Accuracy {rounded_catboost_ac * 100}% ")


# Store accuracies and corresponding functions in a dictionary
model_accuracies = {
    'log_ac': log_ac,
    'knn_ac': knn_ac,
    #     'svc_ac': svc_ac,
    'dt_ac': dt_ac,
    'rf_ac': rf_ac,
    'adb_ac': adb_ac,
    'gb_ac': gb_ac,
    'xgb_ac': xgb_ac,
    'catboost': catboost_ac
}
# Find the maximum accuracy and call the corresponding function
max_accuracy_key = max(model_accuracies, key=model_accuracies.get)
max_accuracy_value = model_accuracies[max_accuracy_key]

# Call the corresponding function
if max_accuracy_key == 'log_ac':
    log_reg()
if max_accuracy_key == 'knn_ac':
    knn_reg()
# elif max_accuracy_key == 'svc_ac':
#     svc_reg()
elif max_accuracy_key == 'dt_ac':
    dt_reg()
elif max_accuracy_key == 'rf_ac':
    rf_reg()
elif max_accuracy_key == 'adb_ac':
    adb_boosting()
elif max_accuracy_key == 'xgb_ac':
    xg_boost()
elif max_accuracy_key == 'catboost_ac':
    cat_boost()
else:
    gb_boosting()

# Find the maximum accuracy
max_accuracy_value = max(model_accuracies.values())

# Find all models with the maximum accuracy
best_models = [key for key, value in model_accuracies.items() if value == max_accuracy_value]

# Handle the case where there are multiple best models (e.g., by printing them)
if len(best_models) == 1:
    best_model = best_models[0]
    if best_model == 'log_ac':
        log_reg()
    elif best_model == 'knn_ac':
        knn_reg()
    #     elif best_model == 'svc_ac':
    #         svc_reg()
    elif best_model == 'dt_ac':
        dt_reg()
    elif best_model == 'rf_ac':
        rf_reg()
    elif best_model == 'adb_ac':
        adb_boosting()
    elif best_model == 'xgb_ac':
        xg_boost()
    elif best_model == 'catboost_ac':
        cat_boost()
    else:
        gb_boosting()
else:
    print(f"Multiple models ({', '.join(best_models)}) have the same maximum accuracy.")
    # Handle the case of multiple best models as needed
