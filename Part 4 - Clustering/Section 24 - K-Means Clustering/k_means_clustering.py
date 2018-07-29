import matplotlib
import matplotlib.pyplot as plot
import pandas as pd

from sklearn.cluster import KMeans

matplotlib.use("TkAGG")

# Importing data set
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, 3:5].values

# Using elbow method to find optimal clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++")
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plot.plot(range(1, 11), wcss)
plot.title("The Elbow Method")
plot.xlabel("Number of Clusters")
plot.ylabel("WCSS")
plot.show()

# Applying k-means
kmeans = KMeans(n_clusters=5, init="k-means++")
y_kmeans = kmeans.fit_predict(X)

# Visualizing clusters
plot.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c="red", label="Target")
plot.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c="blue", label="Standard")
plot.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c="green", label="Careful")
plot.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c="yellow", label="Sensible")
plot.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, c="purple", label="Careless")
plot.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c="orange", label="Centroids")
plot.title("Clusters of clients")
plot.xlabel("Annual income (k$)")
plot.ylabel("Spending score (1-100)")
plot.legend()
plot.show()
