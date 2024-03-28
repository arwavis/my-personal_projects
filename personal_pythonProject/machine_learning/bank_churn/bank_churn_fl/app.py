from flask import Flask, render_template, request
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import warnings

app = Flask(__name__)

# Read Data
data = pd.read_csv("/Users/aravindv/Documents/code/github/my-personal_projects/my-personal_projects"
                   "/personal_pythonProject/machine_learning/bank_churn/train.csv")


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

    # Save model
    model_filename = f"{model_name}_model.joblib"
    joblib.dump(model, model_filename)
    print(f"{model_name} model saved to {model_filename}")

    return {'model_name': model_name, 'accuracy': accuracy, 'model_filename': model_filename}


# Logistic Regression
log_model = LogisticRegressionCV()
log_result = train_evaluate_model(log_model, "Logistic_Regression")

# K-Nearest Neighbors
knn_model = KNeighborsClassifier()
k_values = list(range(1, 10))
knn_param_grid = {'n_neighbors': k_values}
knn_grid = GridSearchCV(knn_model, knn_param_grid, cv=5, scoring='accuracy')
knn_result = train_evaluate_model(knn_grid, "K_Nearest_Neighbors")

# Decision Tree
dt_model = DecisionTreeClassifier()
dt_result = train_evaluate_model(dt_model, "Decision_Tree")

# Random Forest
rf_model = RandomForestClassifier(n_estimators=10, random_state=101)
rf_result = train_evaluate_model(rf_model, "Random_Forest")

# Ada Boost
adb_model = AdaBoostClassifier(n_estimators=15)
adb_result = train_evaluate_model(adb_model, "Ada_Boost")

# Gradient Boost
gb_model = GradientBoostingClassifier()
gb_param_grid = {"n_estimators": [1, 5, 10], 'max_depth': [3, 4, 5, 6]}
gb_grid = GridSearchCV(gb_model, gb_param_grid)
gb_result = train_evaluate_model(gb_grid, "Gradient_Boost")

# XG Boost
xgb_model = XGBClassifier()
xgb_param_grid = {'n_estimators': [1, 5, 10, 20], 'max_depth': [3, 4, 5, 6]}
xgb_grid = GridSearchCV(xgb_model, xgb_param_grid)
xgb_result = train_evaluate_model(xgb_grid, "XG_Boost")

# Cat Boost
catboost_model = CatBoostClassifier()
catboost_param_grid = {'n_estimators': [1, 5, 10, 20], 'depth': [3, 4, 5, 6]}
catboost_grid = GridSearchCV(catboost_model, catboost_param_grid)
catboost_result = train_evaluate_model(catboost_grid, "Cat_Boost")

# Find the model with the highest accuracy
all_results = [log_result, knn_result, dt_result, rf_result, adb_result, gb_result, xgb_result, catboost_result]
best_model_result = max(all_results, key=lambda x: x['accuracy'])
best_model_name = best_model_result['model_name']
print(f"\nBest Model: {best_model_name}")

# Load the best model for predictions
best_model = joblib.load(best_model_result['model_filename'])


# Predicting on New Data
def predict_on_new_data(input_data):
    scaled_new_data = scaler.transform(input_data)
    prediction = best_model.predict(scaled_new_data)[0]
    return "The customer will continue." if prediction == 0 else "The customer will exit."


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Preprocess the input data
    input_data = {
        'CreditScore': float(request.form['CreditScore']),
        'Geography': int(request.form['Geography']),
        'Gender': int(request.form['Gender']),
        'Age': float(request.form['Age']),
        'Tenure': float(request.form['Tenure']),
        'Balance': float(request.form['Balance']),
        'NumOfProducts': float(request.form['NumOfProducts']),
        'HasCrCard': float(request.form['HasCrCard']),
        'IsActiveMember': float(request.form['IsActiveMember']),
        'EstimatedSalary': float(request.form['EstimatedSalary'])
    }

    new_data = pd.DataFrame(input_data, index=[1])

    # Make prediction
    prediction_result = predict_on_new_data(new_data)

    return render_template('index.html', prediction=prediction_result)


if __name__ == '__main__':
    app.run(debug=True)
