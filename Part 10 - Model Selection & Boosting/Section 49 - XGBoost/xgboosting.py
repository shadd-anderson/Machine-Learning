import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier

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

# Fitting XGBoost to the training set
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the test set results
y_pred = classifier.predict(X_test)

# Making the confusion matrix (indicator of success)
cm = confusion_matrix(y_test, y_pred)

# Applying k-fold cross validation
accuracies = cross_val_score(classifier, X_train, y=y_train, cv=10)
accuracies.mean()
accuracies.std()
