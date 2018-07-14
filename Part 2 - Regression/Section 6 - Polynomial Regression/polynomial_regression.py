import matplotlib.pyplot as plot
import numpy as np
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
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualizing the linear regression result
plot.scatter(X, y, color="red")
plot.plot(X, lin_reg.predict(X), color="blue")
plot.title("Truth or bluff (Linear Regression)")
plot.xlabel("Position level")
plot.ylabel("Salary")
plot.show()

# Visualizing the polynomial regression result
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plot.scatter(X, y, color="red")
plot.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color="blue")
plot.title("Truth or bluff (Polynomial Regression)")
plot.xlabel("Position level")
plot.ylabel("Salary")
plot.show()
