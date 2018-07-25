import matplotlib
import matplotlib.pyplot as plot
import numpy as np
import pandas as pd

from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

matplotlib.use('TkAGG')

# Importing data set and setting independent/dependent variables
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting dataset into Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature scaling (to ensure there are no funky issues with data processing)
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting classifier to training set
classifier = DecisionTreeClassifier(criterion="entropy")
classifier.fit(X_train, y_train)

# Predicting the test set results
y_pred = classifier.predict(X_test)

# Making the confusion matrix (indicator of success)
cm = confusion_matrix(y_test, y_pred)


# Visualizing the results
def plot_regression_results(x_set, y_set, title):
    x1, x2 = np.meshgrid(np.arange(start=x_set[:, 0].min() - 0.5, stop=x_set[:, 0].max() + 0.5, step=0.01),
                         np.arange(start=x_set[:, 1].min() - 0.5, stop=x_set[:, 1].max() + 0.5, step=0.01))
    plot.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
                  alpha=0.75, cmap=ListedColormap(('red', 'green')))
    for i, j in enumerate(np.unique(y_set)):
        plot.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], c=ListedColormap(('red', 'green'))(i), label=j)

    plot.title(title)
    plot.xlabel('Age')
    plot.ylabel('Estimated Salary')
    plot.legend()
    plot.show()


plot_regression_results(X_train, y_train, "Classifier (Training Set)")
plot_regression_results(X_test, y_test, "Classifier (Test Set)")