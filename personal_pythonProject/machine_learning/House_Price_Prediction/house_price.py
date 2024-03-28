import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV
from joblib import dump, load

# Reading the Dataset
df = pd.read_csv("house_data.csv")

# Data Processing
# Removing the columns that are not required for the analysis
# Dropping the fields Date, Street, Satezip, country.

df.drop(df.columns[[0, 14, 16, 17]], axis=1, inplace=True)

# Converting categorical variables of the dataset into numerical variables - using ONE HOT ENCODING technique

df['city'] = df['city'].apply(
    {'Shoreline': 0, 'Seattle': 1, 'Kent': 2, 'Bellevue': 3, 'Redmond': 4, 'Maple Valley': 5, 'North Bend': 6,
     'Lake Forest Park': 7,
     'Sammamish': 8, 'Auburn': 9, 'Des Moines': 10, 'Bothell': 11, 'Federal Way': 12, 'Kirkland': 13, 'Issaquah': 14,
     'Woodinville': 15, 'Normandy Park': 16, 'Fall City': 17, 'Renton': 18, 'Carnation': 19, 'Snoqualmie': 20,
     'Duvall': 21, 'Burien': 22, 'Covington': 23, 'Inglewood-Finn Hill': 24, 'Kenmore': 25, 'Newcastle': 26,
     'Mercer Island': 27,
     'Black Diamond': 28, 'Ravensdale': 29, 'Clyde Hill': 30, 'Algona': 31, 'Skykomish': 32, 'Tukwila': 33,
     'Vashon': 34,
     'Yarrow Point': 35, 'SeaTac': 36, 'Medina': 37, 'Enumclaw': 38, 'Snoqualmie Pass': 39, 'Pacific': 40,
     'Beaux Arts Village': 41,
     'Preston': 42, 'Milton': 43}.get)

# Dividing Dataset into Dependent and Independent columns
X = df.drop('price', axis=1)
y = df['price']

# Splitting the dataset into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Splitting the dataset into training and testing set using polynomial feature
polynomial_converter = PolynomialFeatures(degree=1, include_bias=False)
poly_features = polynomial_converter.fit_transform(X)
P_X_train, P_X_test, P_y_train, P_y_test = train_test_split(poly_features, y, test_size=0.3, random_state=101)

scaler = StandardScaler()
scaler.fit(P_X_train)
P_X_train = scaler.transform(P_X_train)
P_X_test = scaler.transform(P_X_test)
# Choosing a scoring: https://scikit-learn.org/stable/modules/model_evaluation.html
# Negative RMSE so all metrics follow convention "Higher is better"
# See all options: sklearn.metrics.SCORERS.keys()
# -----------------------------------------------------------------------------------Linear Regression-----------------------------------------------------------------------------------#

lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
lin_test_predictions = lin_model.predict(X_test)
lin_MAE = mean_absolute_error(y_test, lin_test_predictions)
lin_MSE = mean_squared_error(y_test, lin_test_predictions)
lin_RMSE = np.sqrt(lin_MSE)
print("Calculating Errors based on Linear Regression")
print(f"lin_MAE= {lin_MAE},lin_MSE= {lin_MSE}, lin_RMSE= {lin_RMSE}")

# -----------------------------------------------------------------------------------Polynomial Regression-----------------------------------------------------------------------------------#

p_final_poly_converter = PolynomialFeatures(degree=1, include_bias=False)
p_final_model = LinearRegression()
p_final_model.fit(p_final_poly_converter.fit_transform(X), y)
poly_test_predictions = p_final_model.predict(P_X_test)
poly_MAE = mean_absolute_error(P_y_test, poly_test_predictions)
poly_MSE = mean_squared_error(P_y_test, poly_test_predictions)
poly_RMSE = np.sqrt(poly_MSE)
print("Calculating Errors based on Polynomial Regression")
print(f"poly_MAE= {poly_MAE}, poly_MSE= {poly_MSE}, poly_RMSE= {poly_RMSE}")

# -----------------------------------------------------------------------------------Ridge Regression-----------------------------------------------------------------------------------#

r_final_poly_converter = PolynomialFeatures(degree=1, include_bias=False)
ridge_cv_model = RidgeCV(alphas=(0.1, 1.0, 10.0), scoring='neg_mean_absolute_error')
ridge_cv_model.fit(r_final_poly_converter.fit_transform(X), y)
# ridge_cv_model.fit(P_X_train, P_y_train)
ridg_test_predictions = ridge_cv_model.predict(P_X_test)
ridg_MAE = mean_absolute_error(P_y_test, ridg_test_predictions)
ridg_MSE = mean_squared_error(P_y_test, ridg_test_predictions)
ridg_RMSE = np.sqrt(ridg_MSE)
print("Calculating Errors based on Ridge Regression")
print(f"ridg_MAE= {ridg_MAE}, ridg_MSE= {ridg_MSE}, ridg_RMSE= {ridg_RMSE}")

# -----------------------------------------------------------------------------------Lasso Regression-----------------------------------------------------------------------------------#

l_final_poly_converter = PolynomialFeatures(degree=1, include_bias=False)
lasso_cv_model = LassoCV(eps=0.1, n_alphas=100, cv=5)
lasso_cv_model.fit(l_final_poly_converter.fit_transform(X), y)
# lasso_cv_model.fit(P_X_train, P_y_train)
lass_test_predictions = lasso_cv_model.predict(P_X_test)
lass_MAE = mean_absolute_error(P_y_test, lass_test_predictions)
lass_MSE = mean_squared_error(P_y_test, lass_test_predictions)
lass_RMSE = np.sqrt(lass_MSE)
print("Calculating Errors based on Lasso Regression")
print(f"lass_MAE= {lass_MAE}, lass_MSE= {lass_MSE}, lass_RMSE= {lass_RMSE}")

# -----------------------------------------------------------------------------------ElasticNet Regression-----------------------------------------------------------------------------------#

e_final_poly_converter = PolynomialFeatures(degree=1, include_bias=False)
elastic_model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], tol=0.01)
elastic_model.fit(e_final_poly_converter.fit_transform(X), y)
# elastic_model.fit(P_X_train, P_y_train)
elas_test_predictions = elastic_model.predict(P_X_test)
elas_MAE = mean_absolute_error(P_y_test, elas_test_predictions)
elas_MSE = mean_squared_error(P_y_test, elas_test_predictions)
elas_RMSE = np.sqrt(elas_MSE)
print("Calculating Errors based on ElasticNet Regression")
print(f"elas_MAE= {elas_MAE}, elas_MSE= {elas_MSE}, elas_RMSE= {elas_RMSE}")

# Prediction on New Data
cost = {'bedrooms': 3.0, 'bathrooms': 2.00, 'sqft_living': 1570, 'sqft_lot': 7500, 'floors': 2.0,
        'waterfront': 0,
        'view': 4, 'condition': 5,
        'sqft_above': 3560, 'sqft_basement': 300, 'yr_built': 1932, 'yr_renovated': 2007, 'city': 2}
index = [1]  # serial number
my_data = pd.DataFrame(cost, index)


def l_p():
    lp_my_data_price = lin_model.predict(my_data)
    lp_rounded_price = np.round(lp_my_data_price, 2)
    print(f" The predicted price for the given data using Linear Regression is :{lp_rounded_price}")


def p_p():
    dump(p_final_model, 'house_predict_poly_model.joblib')
    dump(p_final_poly_converter, 'house_predict_converter.joblib')
    pp_loaded_poly = load('house_predict_converter.joblib')
    pp_loaded_model = load('house_predict_poly_model.joblib')
    cost_poly = pp_loaded_poly.transform(my_data)
    pp_my_data_price = p_final_model.predict(cost_poly)
    pp_rounded_price = np.round(pp_my_data_price, 2)
    print(f" The predicted price for the given data using Polynomial Regression is :{pp_rounded_price}")


def r_p():
    r_my_data_price = ridge_cv_model.predict(my_data.values)
    r_rounded_price = np.round(r_my_data_price, 2)
    print(f" The predicted price for the given data using Ridge Regression is :{r_rounded_price}")


def la_p():
    la_my_data_price = lasso_cv_model.predict(my_data.values)
    la_rounded_price = np.round(la_my_data_price, 2)
    print(f" The predicted price for the given  data using Lasso Regression is :{la_rounded_price}")


def e_p():
    e_my_data_price = elastic_model.predict(my_data.values)
    e_rounded_price = np.round(e_my_data_price, 2)
    print(f" The predicted price for the given data is :{e_rounded_price}")


# Condition to check which algorithm to use
if lin_RMSE <= poly_RMSE and lin_RMSE <= ridg_RMSE and lin_RMSE <= lass_RMSE and lin_RMSE <= elas_RMSE:
    print(f"Using Algorithm : Linear Regression with an Erorr {lin_MAE}")
    l_p()
elif poly_RMSE <= lin_RMSE and poly_RMSE <= ridg_RMSE and poly_RMSE <=  lass_RMSE and poly_RMSE <= elas_RMSE:
    print(f"Using Algorithm : Polynomial Regression with an Error {poly_MAE}")
    p_p()
elif ridg_RMSE <= lin_RMSE and ridg_RMSE <= poly_RMSE and ridg_RMSE <= lass_RMSE and ridg_RMSE <= elas_RMSE:
    print(f"Using Algorithm Ridge Regression with an Error {ridg_MAE}")
    r_p()
elif lass_RMSE <= lin_RMSE and lass_RMSE <=poly_RMSE and lass_RMSE <= ridg_RMSE and lass_RMSE <= elas_RMSE:
    print(f"Using Algorithm :Lasso Regression with an Error {lass_MAE}")
    la_p()
else:
    print(f"Using Algorithm :ElasticNet with an Error {elas_MAE}")
    e_p()

