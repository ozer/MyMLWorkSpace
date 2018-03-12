import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("Mall_Customers.csv")

X = dataset.iloc[:, [3, 4]].values

from scipy.cluster import hierarchy as sch

# take care the largest vertical and check the max number of vertical lines on a horizontal line
dendogram = sch.dendrogram(sch.linkage(X, method='ward', metric='euclidean'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')
plt.show()

# fitting the hierarchical cluster
# when it comes to large data sets, prefer KMeans
from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')

y_hc = hc.fit_predict(X)

# Visualizing the cluster


plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s=100, color='red', label='Careful')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s=100, color='blue', label='Standard')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s=100, color='green', label='Target')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s=100, color='cyan', label='Careless')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s=100, color='magenta', label='Sensible')
plt.title("Clusters of the clients")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()
