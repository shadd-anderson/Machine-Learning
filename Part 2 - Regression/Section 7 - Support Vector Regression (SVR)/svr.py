import matplotlib
import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVR
matplotlib.use('TkAGG')

# Importing data set and setting independent/dependent variables
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1:].values

# # Splitting dataset into Training set and Test set
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling (to ensure there are no funky issues with data processing)
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# Fitting SVR to the dataset
regressor = SVR(kernel="rbf")
regressor.fit(X, y)

# Predicting a new result
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

# Visualizing the SVR result
plot.scatter(X, y, color="red")
plot.plot(X, regressor.predict(X), color="blue")
plot.title("Truth or bluff (SVR)")
plot.xlabel("Position level")
plot.ylabel("Salary")
plot.show()
