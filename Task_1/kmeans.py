from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mode

iris = load_iris()
x = iris.data
y = iris.target
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

km = KMeans(n_clusters=3, random_state=42)
labels = km.fit_predict(x_scaled)

new_labels = np.zeros_like(labels)
for i in range(3):
    mask = (labels == i)
    new_labels[mask] = mode(y[mask])[0]

print("accuracy:", accuracy_score(y, new_labels))
print("precision:", precision_score(y, new_labels, average='macro'))
print("recall:", recall_score(y, new_labels, average='macro'))
print("F1:", f1_score(y, new_labels, average='macro'))

tsne = TSNE(n_components=2, random_state=42)
x_2d = tsne.fit_transform(x_scaled)

plt.figure()
plt.scatter(x_2d[:,0], x_2d[:,1], c=labels)
plt.title("KMeans")
plt.show()
plt.figure()
plt.scatter(x_2d[:,0], x_2d[:,1], c=y)
plt.title("Prawdziwe klasy")
plt.show()