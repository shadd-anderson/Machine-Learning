import numpy as np
import matplotlib
import matplotlib.pyplot as plot
import pandas as pd
import scipy.cluster.hierarchy as sch

from sklearn.cluster import AgglomerativeClustering

matplotlib.use("TkAGG")

dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:, [3, 4]].values

# Using dendrogram to find optimal number of clusters
dendrogram = sch.dendrogram(sch.linkage(X, method="ward"))
plot.title("Dendrogram")
plot.xlabel("Customers")
plot.ylabel("Euclidean Distances")
plot.show()

# Fitting hierarchical clustering to the dataset
hc = AgglomerativeClustering(n_clusters=5)
y_hc = hc.fit_predict(X)

# Visualizing the clusters
plot.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s=100, c="red", label="Careful")
plot.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s=100, c="blue", label="Standard")
plot.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s=100, c="green", label="Target")
plot.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s=100, c="yellow", label="Careless")
plot.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s=100, c="purple", label="Sensible")
plot.title("Clusters of clients")
plot.xlabel("Annual income (k$)")
plot.ylabel("Spending score (1-100)")
plot.legend()
plot.show()
