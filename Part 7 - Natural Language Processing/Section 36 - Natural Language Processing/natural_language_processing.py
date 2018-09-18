import matplotlib
import matplotlib.pyplot as plot
import nltk
import numpy as np
import pandas as pd
import re

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter="\t", quoting=3)

# Cleaning the texts
# nltk.download("stopwords")
corpus = []
ps = PorterStemmer()
for i in range(0, 1000):
    review = re.sub("[^a-zA-Z]", " ", dataset['Review'][i])
    review = review.lower().split()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words("english"))]
    review = " ".join(review)
    corpus.append(review)

# Creating bag of words
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting dataset into Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fitting classifier to training set
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the test set results
y_pred = classifier.predict(X_test)

# Making the confusion matrix (indicator of success)
cm = confusion_matrix(y_test, y_pred)


def predict(new_review):
    new_review = re.sub("[^a-zA-Z]", " ", new_review)
    new_review = new_review.lower().split()
    new_review = [ps.stem(word) for word in new_review if word not in set(stopwords.words("english"))]
    new_review = " ".join(new_review)
    new_review = [new_review]
    new_review = cv.transform(new_review).toarray()
    if classifier.predict(new_review)[0] == 1:
        return "Positive"
    else:
        return "Negative"
