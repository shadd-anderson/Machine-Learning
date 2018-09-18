import keras
import numpy as np
import pandas as pd

from keras.layers import Dense
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# Importing data set and setting independent/dependent variables
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# Encoding categorical data
labelendcoder_X_1 = LabelEncoder()
X[:, 1] = labelendcoder_X_1.fit_transform(X[:, 1])
labelendcoder_X_2 = LabelEncoder()
X[:, 2] = labelendcoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting dataset into Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature scaling (to ensure there are no funky issues with data processing)
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Initializing the ANN
classifier = Sequential()

# Adding the layers
classifier.add(Dense(units=6, kernel_initializer="uniform", activation="relu", input_dim=11))
classifier.add(Dense(units=6, kernel_initializer="uniform", activation="relu"))
classifier.add(Dense(units=1, kernel_initializer="uniform", activation="sigmoid"))

# Compiling ANN
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

# Predicting the test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the confusion matrix (indicator of success)
cm = confusion_matrix(y_test, y_pred)
