import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import numpy as np
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
true_labels = iris.target
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(data)
predicted_labels = np.zeros_like(kmeans.labels_)
for i in range(3): mask = (kmeans.labels_ == i)
most_common_label = np.bincount(true_labels[mask]).argmax()
predicted_labels[mask] = most_common_label
accuracy = accuracy_score(true_labels, predicted_labels)
print(f'Accuracy of K-Means clustering: {accuracy:.2f}')
plt.scatter(data['sepal length (cm)'], data['sepal width (cm)'],c=kmeans.labels_,cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red',
marker='X', label='Centroids')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('K-means Clustering on Iris Dataset')
plt.legend()
plt.show()
