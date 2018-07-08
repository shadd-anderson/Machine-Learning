import matplotlib.pyplot as plot
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Importing data set and setting independent/dependent variables
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting dataset into Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# Fitting Simple Linear Regression to the training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the test set results
y_pred = regressor.predict(X_test)

# Visualizing the training set results
plot.scatter(X_train, y_train, color="red")
plot.plot(X_train, regressor.predict(X_train), color="blue")
plot.title("Salary vs Experience (Training set)")
plot.xlabel("Years of experience")
plot.ylabel("Salary")
plot.show()

# Visualizing the test set results
plot.scatter(X_test, y_test, color="red")
plot.plot(X_train, regressor.predict(X_train), color="blue")
plot.title("Salary vs Experience (Test set)")
plot.xlabel("Years of experience")
plot.ylabel("Salary")
plot.show()
