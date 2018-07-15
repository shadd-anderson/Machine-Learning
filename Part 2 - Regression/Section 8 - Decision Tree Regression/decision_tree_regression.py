import matplotlib
import matplotlib.pyplot as plot
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
matplotlib.use('TkAGG')

# Importing data set and setting independent/dependent variables
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1:].values

# Fitting Decision Tree Regression to the dataset
regressor = DecisionTreeRegressor()
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict(6.5)

# Visualizing the Decision Tree Regression result
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)
plot.scatter(X, y, color="red")
plot.plot(X_grid, regressor.predict(X_grid), color="blue")
plot.title("Truth or bluff (Decision Tree Regression)")
plot.xlabel("Position level")
plot.ylabel("Salary")
plot.show()
