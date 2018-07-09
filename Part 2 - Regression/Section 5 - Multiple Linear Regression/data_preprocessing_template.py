import numpy as np
import matplotlib.pyplot as plot
import pandas as pd

from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Importing data set and setting independent/dependent variables
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1:].values

# # Taking care of missing data
# imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
# imputer = imputer.fit(X[:, 1:])
# X[:, 1:] = imputer.transform(X[:, 1:])

# # Encoding categorical data
# # --Encodes country category from string to integer
# labelendcoder_X = LabelEncoder()
# X[:, 0] = labelendcoder_X.fit_transform(X[:, 0])
# # --Transforms country integers to 3 "true or false" columns for each country
# # --This is because there is no hierarchy to the countries, they're simply categories
# onehotencoder = OneHotEncoder(categorical_features=[0])
# X = onehotencoder.fit_transform(X).toarray()
# # --Encodes yes or no to 1 or 0
# labelendcoder_y = LabelEncoder()
# y = labelendcoder_y.fit_transform(y)

# Splitting dataset into Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Feature scaling (to ensure there are no funky issues with data processing)
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)
