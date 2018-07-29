import numpy as np
import matplotlib
import matplotlib.pyplot as plot
import pandas as pd

import scipy.cluster.hierarchy as sch

matplotlib.use("TkAGG")

dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:, [3, 4]].values

# Using dendrogram to find optimal number of clusters
dendrogram = sch.dendrogram(sch.linkage(X, method="ward"))
plot.title("Dendrogram")
plot.xlabel("Customers")
plot.ylabel("Euclidean Distances")
plot.show()
