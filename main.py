import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def kmeans(X, k, max_iter=100):
    n, d = X.shape
    centroids = X[np.random.choice(n, k, replace=False), :]
    labels = np.zeros(n)
    for _ in range(max_iter):
        distances = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=-1)
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for j in range(k):
            centroids[j] = np.mean(X[labels == j], axis=0)
    return labels, centroids


def som(X, som_shape, sigma=1.0, learning_rate=0.5, max_iter=100):
    n, d = X.shape
    som_size = som_shape[0] * som_shape[1]
    som = np.random.randn(som_size, d)
    for _ in range(max_iter):
        x = X[np.random.randint(n)]
        distances = np.linalg.norm(som - x, axis=1)
        winner = np.argmin(distances)
        winner_row = winner // som_shape[1]
        winner_col = winner % som_shape[1]
        distance_from_winner = np.linalg.norm(
            np.indices(som_shape).reshape(2, -1).T - np.array([winner_row, winner_col]),
            axis=1
        )
        neighborhood = np.exp(-distance_from_winner ** 2 / (2 * sigma ** 2))
        som += learning_rate * neighborhood[:, None] * (x - som)
    som_indices = np.arange(som_size).reshape(som_shape)
    som_labels = np.argmin(
        np.linalg.norm(X[:, None, None] - som[None, :, :], axis=-1),
        axis=2
    )
    som_labels = som_indices[som_labels // som_shape[1], som_labels % som_shape[1]].reshape(-1)
    return som_labels, som


# Load the Kaggle dataset
data0 = pd.read_csv('MAISTO_DUOMENYS.csv', sep=';')
data = data0.fillna(0)

# Extract the features from the dataset
#feature_columns = ['Water_(g)', 'Energ_Kcal', 'Protein_(g)' ,'Lipid_Tot_(g)', 'Carbohydrt_(g)',
#                   'Fiber_TD_(g)', 'Sugar_Tot_(g)', 'Iron_(mg)', 'Cholestrl_(mg)']
# feature_columns = ['Water_(g)', 'Protein_(g)' , 'Cholestrl_(mg)'] #  0.55 su 5 K, 0.6 su K=3
#feature_columns = [ 'Protein_(g)' , 'Energ_Kcal', 'Iron_(mg)']
feature_columns = ['Carbohydrt_(g)', 'Sugar_Tot_(g)']

# # Generate all combinations
# max_k = -10
# max_s = -10
# max_k_id = []
# max_s_id = []
# max_sum = -10
# max_sum_id = []
# for r in range(2, len(feature_columns) + 1):
#     combinations = list(itertools.combinations(feature_columns, r))
#     print(f"Combinations of size {r}:")
#     for combination in combinations:
#         print(combination)
#         X = data[list(combination)].values
#
#         # Normalize the data
#         scaler = StandardScaler()
#         X = scaler.fit_transform(X)
#
#         # K-means clustering
#         k = 3
#         kmeans_labels, kmeans_centroids = kmeans(X, k)
#         if len(np.unique(kmeans_labels)) < 2:
#             print("Skipping - Only 1 cluster found.")
#             continue
#         kmeans_silhouette = silhouette_score(X, kmeans_labels)
#         print("K-Means: ", kmeans_silhouette)
#         if kmeans_silhouette > max_k:
#             max_k = kmeans_silhouette
#             max_k_id = combination
#
#         # SOM clustering
#         som_shape = (3, 1)
#         sigma = 1.0
#         learning_rate = 0.5
#         som_labels, som1 = som(X, som_shape, sigma, learning_rate)
#         if len(np.unique(som_labels)) < 2:
#             print("Skipping - Only 1 cluster found.")
#             continue
#         som_silhouette = silhouette_score(X, som_labels)
#         print("SOM: ", som_silhouette)
#         if som_silhouette > max_s:
#             max_s = som_silhouette
#             max_s_id = combination
#         if som_silhouette + kmeans_silhouette > max_sum:
#             max_sum = som_silhouette + kmeans_silhouette
#             max_sum_id = combination
#     print()
# print(max_k)
# print(max_k_id)
# print()
# print(max_s)
# print(max_s_id)
# print()
# print(max_sum)
# print(max_sum_id)
# print()

X = data[feature_columns].values

# Normalize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# K-means clustering
k = 3
kmeans_labels, kmeans_centroids = kmeans(X, k)
kmeans_silhouette = silhouette_score(X, kmeans_labels)

# SOM clustering
som_shape = (3, 1)
sigma = 1.0
learning_rate = 0.5
som_labels, som = som(X, som_shape, sigma, learning_rate)
som_silhouette = silhouette_score(X, som_labels)

# Plot results
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(X[:, 0], X[:, 1], c=kmeans_labels)
axes[0].scatter(kmeans_centroids[:, 0], kmeans_centroids[:, 1], s=100, marker='x', c='black')
axes[0].set_title(f"K-means clustering (Silhouette score: {kmeans_silhouette:.2f})")
axes[1].scatter(X[:, 0], X[:, 1], c=som_labels)
axes[1].scatter(som[:, 0], som[:, 1], s=100, marker='x', c='black')
axes[1].set_title(f"SOM clustering (Silhouette score: {som_silhouette:.2f})")
plt.show()