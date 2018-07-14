import pandas as pd

# Importing data set and setting independent/dependent variables
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1:].values

# # Splitting dataset into Training set and Test set
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
