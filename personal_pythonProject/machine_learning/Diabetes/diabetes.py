# Import Libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Reading the Dataset
data = pd.read_csv("/Users/aravindv/Documents/code/github/MyMLMinds/Complete_MLpredPycode/Diabetes/diabetes.csv")
# print(data.head())

# Dividing Dataset into Dependent and Independent columns
X= data.drop('Outcome',axis=1)
y=data['Outcome']

# Splitting the dataset into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)
scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

# -----------------------------------------------------------------------------------Logistic Regression

log_model = LogisticRegressionCV()
log_model.fit(scaled_X_train, y_train)
log_pred = log_model.predict(scaled_X_test)
log_cm = confusion_matrix(y_test, log_pred)
log_ac = accuracy_score(y_test, log_pred)
rounded_log_ac = np.round(float(log_ac), 2)
print(f"Accuracy (Logistic Regression): {rounded_log_ac * 100}")
# print("Confusion Matrix {0}".format(log_conf))
print("Confusion Matrix: ")
print(log_cm)
# -----------------------------------------------------------------------------------KNN - K Nearest Neighbours
knn = KNeighborsClassifier()
operations = [('scaler', scaler), ('knn', knn)]
pipe = Pipeline(operations)
k_values = list(range(1, 30))
knn_param_grid = {'knn__n_neighbors': k_values}
full_cv_classifier = GridSearchCV(pipe, knn_param_grid, cv=5, scoring='accuracy')
full_cv_classifier.fit(scaled_X_train, y_train)
full_cv_classifier.best_estimator_.get_params()
knn_pred = full_cv_classifier.predict(scaled_X_test)
knn_cm = confusion_matrix(y_test, knn_pred)
knn_ac = accuracy_score(y_test, knn_pred)
rounded_knn_ac = np.round(float(knn_ac), 2)
print(f"Accuracy (KNN): {rounded_knn_ac * 100}")
print("Confusion Matrix: ")
print(knn_cm)

# -----------------------------------------------------------------------------------Support Vector Machine
svc = SVC(class_weight='balanced')
svc_param_grid = {'C': [0.001, 0.01, 0.1, 0.5, 1], 'gamma': ['scale', 'auto']}
svc_grid = GridSearchCV(svc, svc_param_grid)
svc_grid.fit(scaled_X_train, y_train)
svc_grid_pred = svc_grid.predict(scaled_X_test)
svc_cm = confusion_matrix(y_test, svc_grid_pred)
svc_ac = accuracy_score(y_test, svc_grid_pred)
rounded_svc_ac = np.round(float(svc_ac), 2)
print(f"Accuracy (SVM): {rounded_svc_ac * 100}")
print("Confusion Matrix: ")
print(svc_cm)

# -----------------------------------------------------------------------------------Decision Tree Machine
decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(scaled_X_train, y_train)
decision_tree_pred = decision_tree_model.predict(scaled_X_test)
dt_cm = confusion_matrix(y_test, decision_tree_pred)
dt_ac = accuracy_score(y_test, decision_tree_pred)
rounded_dt_ac = np.round(float(dt_ac), 2)
print(f"Accuracy (DTM): {rounded_dt_ac * 100}")
print("Confusion Matrix: ")
print(dt_cm)

# -----------------------------------------------------------------------------------Random Forest Machine
# Use 10 random trees
Random_Forest_model = RandomForestClassifier(n_estimators=10, random_state=101)
Random_Forest_model.fit(scaled_X_train, y_train)
Random_Forest_preds = Random_Forest_model.predict(scaled_X_test)
rf_cm = confusion_matrix(y_test, Random_Forest_preds)
rf_ac = accuracy_score(y_test, Random_Forest_preds)
rounded_rf_ac = np.round(float(rf_ac), 2)
print(f"Accuracy (RFM): {rounded_rf_ac * 100}")
print("Confusion Matrix: ")
print(rf_cm)

# -----------------------------------------------------------------------------------Boosting - Ada Boost
ada_boost_model = AdaBoostClassifier(n_estimators=15)
ada_boost_model.fit(scaled_X_train, y_train)
ada_boost_preds = ada_boost_model.predict(scaled_X_test)
adb_cm = confusion_matrix(y_test, ada_boost_preds)
adb_ac = accuracy_score(y_test, ada_boost_preds)
rounded_adb_ac = np.round(float(adb_ac), 2)
print(f"Accuracy (A-Boost): {rounded_adb_ac * 100}")
print("Confusion Matrix: ")
print(adb_cm)

# -----------------------------------------------------------------------------------Boosting - Gradient Boost

gb_model = GradientBoostingClassifier()
gb_param_grid = {"n_estimators": [1, 5, 10, 20, 40, 100], 'max_depth': [3, 4, 5, 6]}
gb_grid = GridSearchCV(gb_model, gb_param_grid)
gb_grid.fit(scaled_X_train, y_train)
gb_predictions = gb_grid.predict(scaled_X_test)
gb_cm = confusion_matrix(y_test, gb_predictions)
gb_ac = accuracy_score(y_test, gb_predictions)
rounded_gb_ac = np.round(float(gb_ac), 2)
print(f"Accuracy (G-Boost): {rounded_gb_ac * 100}")
print("Confusion Matrix: ")
print(gb_cm)

# Prediction on New Data
data_new = {'Pregnancies': 6, 'Glucose': 148, 'BloodPressure': 72, 'SkinThickness': 35, 'Insulin': 0,
            'BMI': 33.6, 'DiabetesPedigreeFunction': 0.627, 'Age': 50}
index = [1]  # serial number
my_data = pd.DataFrame(data_new, index)

# Calling some functions for prediction using the above models.
def log_reg():
    medical_details = log_model.predict(my_data.values)
    # print(type(medical_details))
    if medical_details == 0:
        print("Predicted using Logistic Regression: The patient has NO symptoms of Diabetes ")
    else:
        print("Predicted using Logistic Regression: The patient has symptoms of Diabetes ")

    print(f"Above data is calculated with an Accuracy {rounded_log_ac * 100}% ")


def knn_reg():
    medical_details = full_cv_classifier.predict(my_data.values)
    # print(type(medical_details))
    if medical_details == 0:
        print("Predicted using KNN : The patient has NO symptoms of Diabetes ")
    else:
        print("Predicted using KNN: The patient has symptoms of Diabetes ")

    print(f"Above data is calculated with an Accuracy {rounded_knn_ac * 100}% ")


def svc_reg():
    medical_details = decision_tree_model.predict(my_data.values)
    # print(type(medical_details))
    if medical_details == 0:
        print("Predicted using Support Vector Machine : The patient has NO symptoms of Diabetes ")
    else:
        print("Predicted using Support Vector Machine: The patient has symptoms of Diabetes ")

    print(f"Above data is calculated with an Accuracy {rounded_svc_ac * 100}% ")


def dt_reg():
    medical_details = svc_grid.predict(my_data.values)
    # print(type(medical_details))
    if medical_details == 0:
        print("Predicted using Decision Tree : The patient has NO symptoms of Diabetes ")
    else:
        print("Predicted using Decision Tree: The patient has symptoms of Diabetes ")

    print(f"Above data is calculated with an Accuracy {rounded_dt_ac * 100}% ")


def rf_reg():
    medical_details = Random_Forest_model.predict(my_data.values)
    # print(type(medical_details))
    if medical_details == 0:
        print("Predicted using Random Forest : The patient has NO symptoms of Diabetes ")
    else:
        print("Predicted using Random Forest: The patient has symptoms of Diabetes ")

    print(f"Above data is calculated with an Accuracy {rounded_rf_ac * 100}% ")


def adb_boosting():
    medical_details = ada_boost_model.predict(my_data.values)
    # print(type(medical_details))
    if medical_details == 0:
        print("Predicted using Ada Boosting : The patient has NO symptoms of Diabetes ")
    else:
        print("Predicted using Ada Boosting: The patient has symptoms of Diabetes ")

    print(f"Above data is calculated with an Accuracy {rounded_adb_ac * 100}% ")


def gb_boosting():
    medical_details = gb_grid.predict(my_data.values)
    # print(type(medical_details))
    if medical_details == 0:
        print("Predicted using Gradient Boosting : The patient has NO symptoms of Diabetes ")
    else:
        print("Predicted using Gradient Boosting: The patient has symptoms of Diabetes ")

    print(f"Above data is calculated with an Accuracy {rounded_gb_ac * 100}% ")
    
# # Condition to check with Model performs better and predicts the result.
# if log_ac >= knn_ac and log_ac >= svc_ac and log_ac >= dt_ac and log_ac >= rf_ac and log_ac >= adb_ac and log_ac >= gb_ac:
#     log_reg()
# elif knn_ac >= log_ac and knn_ac >= svc_ac and knn_ac >= dt_ac and knn_ac >= rf_ac and knn_ac >= adb_ac and knn_ac >= gb_ac:
#     knn_reg()
# elif svc_ac >= log_ac and svc_cm >= knn_ac and svc_ac >= dt_ac and svc_ac >= rf_ac and svc_ac >= adb_ac and svc_ac >= gb_ac:
#     svc_reg()
# elif dt_ac >= log_ac and dt_ac >= knn_ac and dt_ac >= svc_ac and dt_ac >= rf_ac and dt_ac >= adb_ac and dt_ac >= gb_ac:
#     dt_reg()
# elif rf_ac >= log_ac and rf_ac >= knn_ac and rf_ac >= svc_ac and rf_ac >= dt_ac and rf_ac >= adb_ac and rf_ac >= gb_ac:
#     rf_reg()
# elif adb_ac >= log_ac and adb_ac >= knn_ac and adb_ac >= svc_ac and adb_ac >= dt_ac and adb_ac >= rf_ac and adb_ac >= gb_ac:
#     adb_boosting()
# else:
#     gb_boosting()

#Alternative: instead of using the long conditional statment above.

# Store accuracies and corresponding functions in a dictionary
model_accuracies = {
    'log_ac': log_ac,
    'knn_ac': knn_ac,
    'svc_ac': svc_ac,
    'dt_ac': dt_ac,
    'rf_ac': rf_ac,
    'adb_ac': adb_ac,
    'gb_ac': gb_ac
}

# Find the maximum accuracy and call the corresponding function
max_accuracy_key = max(model_accuracies, key=model_accuracies.get)
max_accuracy_value = model_accuracies[max_accuracy_key]

# Call the corresponding function
if max_accuracy_key == 'log_ac':
    log_reg()
elif max_accuracy_key == 'knn_ac':
    knn_reg()
elif max_accuracy_key == 'svc_ac':
    svc_reg()
elif max_accuracy_key == 'dt_ac':
    dt_reg()
elif max_accuracy_key == 'rf_ac':
    rf_reg()
elif max_accuracy_key == 'adb_ac':
    adb_boosting()
else:
    gb_boosting()

# Handle the case where there are multiple best models
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
    elif best_model == 'svc_ac':
        svc_reg()
    elif best_model == 'dt_ac':
        dt_reg()
    elif best_model == 'rf_ac':
        rf_reg()
    elif best_model == 'adb_ac':
        adb_boosting()
    else:
        gb_boosting()
else:
    print(f"Multiple models ({', '.join(best_models)}) have the same maximum accuracy.")
    # Handle the case of multiple best models as needed