import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import warnings

warnings.filterwarnings("ignore")

# Read Data
data = pd.read_csv("train.csv")

# Data Preprocessing
def preprocess_data(data):
    data = data.drop(["id", "CustomerId", "Surname"], axis=1)
    data['Geography'] = data['Geography'].map({'France': 0, 'Spain': 1, 'Germany': 2})
    data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})
    return data

data = preprocess_data(data)

# Train-Test Split
X = data.drop('Exited', axis=1)
y = data['Exited']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)

# Scaling Data
scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

# Model Training and Evaluation
def train_evaluate_model(model, model_name):
    model.fit(scaled_X_train, y_train)
    predictions = model.predict(scaled_X_test)
    accuracy = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    rounded_accuracy = np.round(accuracy, 2)
    print(f"{model_name} Accuracy: {rounded_accuracy * 100}%")
    print(f"{model_name} Confusion Matrix:")
    print(cm)
    return {'model_name': model_name, 'accuracy': rounded_accuracy, 'confusion_matrix': cm}

# Logistic Regression
log_model = LogisticRegressionCV()
log_result = train_evaluate_model(log_model, "Logistic Regression")

# K-Nearest Neighbors
knn_model = KNeighborsClassifier()
k_values = list(range(1, 10))
knn_param_grid = {'n_neighbors': k_values}
knn_grid = GridSearchCV(knn_model, knn_param_grid, cv=5, scoring='accuracy')
knn_result = train_evaluate_model(knn_grid, "K-Nearest Neighbors")

# Decision Tree
dt_model = DecisionTreeClassifier()
dt_result = train_evaluate_model(dt_model, "Decision Tree")

# Random Forest
rf_model = RandomForestClassifier(n_estimators=10, random_state=101)
rf_result = train_evaluate_model(rf_model, "Random Forest")

# Ada Boost
adb_model = AdaBoostClassifier(n_estimators=15)
adb_result = train_evaluate_model(adb_model, "Ada Boost")

# Gradient Boost
gb_model = GradientBoostingClassifier()
gb_param_grid = {"n_estimators": [1, 5, 10], 'max_depth': [3, 4, 5, 6]}
gb_grid = GridSearchCV(gb_model, gb_param_grid)
gb_result = train_evaluate_model(gb_grid, "Gradient Boost")

# XG Boost
xgb_model = XGBClassifier()
xgb_param_grid = {'n_estimators': [1, 5, 10, 20], 'max_depth': [3, 4, 5, 6]}
xgb_grid = GridSearchCV(xgb_model, xgb_param_grid)
xgb_result = train_evaluate_model(xgb_grid, "XG Boost")

# Cat Boost
catboost_model = CatBoostClassifier()
catboost_param_grid = {'n_estimators': [1, 5, 10, 20], 'depth': [3, 4, 5, 6]}
catboost_grid = GridSearchCV(catboost_model, catboost_param_grid)
catboost_result = train_evaluate_model(catboost_grid, "Cat Boost")

# Predicting on New Data
def predict_on_new_data(model, model_name):
    new_data = pd.DataFrame({'CreditScore': 668, 'Geography': 0, 'Gender': 0, 'Age': 33, 'Tenure': 3,
                             'Balance': 0.0, 'NumOfProducts': 2, "HasCrCard": 1.0, "IsActiveMember": 0.0,
                             "EstimatedSalary": 181449.97}, index=[1])
    scaled_new_data = scaler.transform(new_data)
    prediction = model.predict(scaled_new_data)
    if prediction == 0:
        print(f"Predicted using {model_name}: The customer will continue.")
    else:
        print(f"Predicted using {model_name}: The customer will exit")

# Find the model with the highest accuracy
all_results = [log_result, knn_result, dt_result, rf_result, adb_result, gb_result, xgb_result, catboost_result]
best_model_result = max(all_results, key=lambda x: x['accuracy'])
best_model_name = best_model_result['model_name']
print(f"\nBest Model: {best_model_name}")

# Predict on new data using the best model
if best_model_name == 'K-Nearest Neighbors':
    predict_on_new_data(knn_grid, best_model_name)
elif best_model_name == 'Decision Tree':
    predict_on_new_data(dt_model, best_model_name)
elif best_model_name == 'Random Forest':
    predict_on_new_data(rf_model, best_model_name)
elif best_model_name == 'Ada Boost':
    predict_on_new_data(adb_model, best_model_name)
elif best_model_name == 'Gradient Boost':
    predict_on_new_data(gb_grid, best_model_name)
elif best_model_name == 'XG Boost':
    predict_on_new_data(xgb_grid, best_model_name)
elif best_model_name == 'Cat Boost':
    predict_on_new_data(catboost_grid, best_model_name)
else:
    predict_on_new_data(log_model, best_model_name)
