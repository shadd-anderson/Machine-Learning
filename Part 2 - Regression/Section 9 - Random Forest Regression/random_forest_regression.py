import matplotlib
import matplotlib.pyplot as plot
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
matplotlib.use('TkAGG')

# Importing data set and setting independent/dependent variables
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1:].values

# Fitting Random Forest Regression to the dataset
regressor = RandomForestRegressor(n_estimators=300)
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict(6.5)

# Visualizing the Random Forest Regression result
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)
plot.scatter(X, y, color="red")
plot.plot(X_grid, regressor.predict(X_grid), color="blue")
plot.title("Truth or bluff (Random Forest Regression)")
plot.xlabel("Position level")
plot.ylabel("Salary")
plot.show()
