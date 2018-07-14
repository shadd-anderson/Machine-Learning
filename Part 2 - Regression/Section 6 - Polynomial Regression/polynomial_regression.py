import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Importing data set and setting independent/dependent variables
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1:].values

# # Splitting dataset into Training set and Test set
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fitting linear model
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting polynomial regression
poly_reg = PolynomialFeatures()
X_poly = poly_reg.fit_transform(X, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
