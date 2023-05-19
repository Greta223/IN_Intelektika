import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def calculate_silhouette_coefficient(data, labels):
    num_samples = len(data)
    silhouette_coefficients = np.zeros(num_samples)

    for i in range(num_samples):
        point = data[i]
        cluster_label = labels[i]

        # Calculate the average distance between the point and all other points in its cluster
        intra_cluster_distances = []
        for j in range(num_samples):
            if labels[j] == cluster_label and i != j:
                intra_cluster_distances.append(calculate_distance(point, data[j]))
        avg_intra_cluster_distance = np.mean(intra_cluster_distances)

        # Calculate the average distance between the point and all points in the nearest neighboring cluster
        inter_cluster_distances = []
        for j in range(num_samples):
            if labels[j] != cluster_label:
                inter_cluster_distances.append(calculate_distance(point, data[j]))
        avg_inter_cluster_distance = np.mean(inter_cluster_distances)

        # Calculate the silhouette coefficient for the point
        silhouette_coefficients[i] = (avg_inter_cluster_distance - avg_intra_cluster_distance) / max(
            avg_inter_cluster_distance, avg_intra_cluster_distance)

    return silhouette_coefficients


#Euklido atstumas
def calculate_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


def kmeans(X, k, max_iter=100):
    #apsibreziame n ir d, n=eiluciu kiekis, d=stulpeliu kiekis
    n, d = X.shape
    #atsitiktiniu budu parenkamas k kiekis centroidu
    centroids = X[np.random.choice(n, k, replace=False), :]
    #masyvas tasku klasteriams saugoti (koks taskas priklauso kokiam klasteriui)
    labels = np.zeros(n)

    #apsauga nuo nesibaigiancio ciklo
    for _ in range(max_iter):
        #masyvas saugantis atstumus nuo kiekvieno tasko iki kiekvieno centroido
        distances = np.zeros((n, k))

        # Calculate distances between each sample and centroids
        #einame per visus taskus
        for i in range(n):
            for j in range(k):
                #pasiskaicuoja atstumus nuo kiekvieno centroido
                distances[i, j] = calculate_distance(X[i], centroids[j])

        #taskams priskiriamu klasteriu masyvas
        new_labels = np.zeros(n)

        # Assign labels based on minimum distance
        for i in range(n):
            min_distance = float('inf')
            for j in range(k):
                distance = distances[i, j]
                if distance < min_distance:
                    min_distance = distance
                    new_labels[i] = j

        if np.array_equal(labels, new_labels):
            break

        #atnaujiname priskyrima klasteriui
        labels = new_labels

        #einame per centroidus ir atnaujiname visus centroidu pozicijas
        for j in range(k):
            centroids[j] = np.mean(X[labels == j], axis=0)

    return labels, centroids


def som(X, som_shape, sigma=1.0, learning_rate=0.5, max_iter=100):
    n, d = X.shape
    #klasteriu kiekis
    som_size = som_shape[0] * som_shape[1]
    som = np.random.randn(som_size, d)

    for _ in range(max_iter):
        x = X[np.random.randint(n)]
        distances = np.zeros(som_size)

        # Calculate distances between each SOM node and input sample
        for i in range(som_size):
            distances[i] = calculate_distance(som[i], x)

        winner = 0
        min_distance = distances[0]

        # Find the winner node with minimum distance
        for i in range(1, som_size):
            if distances[i] < min_distance:
                min_distance = distances[i]
                winner = i

        winner_row = winner // som_shape[1]
        winner_col = winner % som_shape[1]

        distance_from_winner = np.zeros(som_size)

        # Calculate distances from the winner node to other nodes in the SOM grid
        for i in range(som_size):
            row = i // som_shape[1]
            col = i % som_shape[1]
            distance_from_winner[i] = calculate_distance(np.array([row, col]), np.array([winner_row, winner_col]))

        neighborhood = np.exp(-distance_from_winner ** 2 / (2 * sigma ** 2))

        som += learning_rate * neighborhood[:, None] * (x - som)

    som_labels = np.zeros(n)

    # Assign labels based on minimum distance to SOM nodes
    for i in range(n):
        distances = np.zeros(som_size)
        for j in range(som_size):
            distances[j] = calculate_distance(X[i], som[j])

        winner = 0
        min_distance = distances[0]

        # Find the winner node with minimum distance
        for j in range(1, som_size):
            if distances[j] < min_distance:
                min_distance = distances[j]
                winner = j

        som_labels[i] = winner

    return som_labels, som


# Load the Kaggle dataset
data0 = pd.read_csv('MAISTO_DUOMENYS.csv', sep=';')
data = data0.fillna(0)

# Extract the features from the dataset
# feature_columns = ['Water_(g)', 'Energ_Kcal', 'Protein_(g)' ,'Lipid_Tot_(g)', 'Carbohydrt_(g)',
#                   'Fiber_TD_(g)', 'Sugar_Tot_(g)', 'Iron_(mg)', 'Cholestrl_(mg)']
# feature_columns = ['Water_(g)', 'Protein_(g)' , 'Cholestrl_(mg)'] #  0.55 su 5 K, 0.6 su K=3
#feature_columns = [ 'Protein_(g)' , 'Energ_Kcal', 'Iron_(mg)']
# feature_columns = ['Carbohydrt_(g)', 'Sugar_Tot_(g)']
feature_columns = ['Protein_(g)', 'Fiber_TD_(g)', 'Cholestrl_(mg)']

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
#         kmeans_silhouette = np.mean(calculate_silhouette_coefficient(X, kmeans_labels))
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
#         som_silhouette = np.mean(calculate_silhouette_coefficient(X, som_labels))
#         print("SOM: ", som_silhouette)
#         if som_silhouette > max_s:
#             max_s = som_silhouette
#             max_s_id = combination
#         if som_silhouette + kmeans_silhouette > max_sum and abs(som_silhouette-kmeans_silhouette) < 0.3:
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
#kmeans_silhouette = silhouette_score(X, kmeans_labels)
kmeans_silhouette = np.mean(calculate_silhouette_coefficient(X, kmeans_labels))
# # SOM clustering
# som_shape = (3, 1)
# sigma = 1.0
# learning_rate = 0.5
# som_labels, som = som(X, som_shape, sigma, learning_rate)
# #som_silhouette = silhouette_score(X, som_labels)
# som_silhouette = np.mean(calculate_silhouette_coefficient(X, som_labels))
# Plot results
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(X[:, 0], X[:, 1], c=kmeans_labels)
axes[0].scatter(kmeans_centroids[:, 0], kmeans_centroids[:, 1], s=100, marker='x', c='black')
axes[0].set_title(f"K-means clustering (Silhouette score: {kmeans_silhouette:.2f})")
# axes[1].scatter(X[:, 0], X[:, 1], c=som_labels)
# axes[1].scatter(som[:, 0], som[:, 1], s=100, marker='x', c='black')
# axes[1].set_title(f"SOM clustering (Silhouette score: {som_silhouette:.2f})")
plt.show()