
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


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


def calculate_inertia(X, centroids, labels):
    inertia = 0
    for i in range(len(X)):
        centroid = centroids[int(labels[i])]
        distance = calculate_distance(X[i], centroid)
        inertia += distance ** 2
    return inertia


def kmeans(X, k, max_iter=100):
    #Apsibreziame n ir d, n=eiluciu kiekis, d=stulpeliu kiekis
    n, d = X.shape
    #Atsitiktiniu budu parenkamas k kiekis centroidu
    centroids = X[np.random.choice(n, k, replace=False), :]
    #Masyvas tasku klasteriams saugoti (koks taskas priklauso kokiam klasteriui)
    labels = np.zeros(n)

    #Apsauga nuo nesibaigiancio ciklo
    for _ in range(max_iter):
        #Masyvas saugantis atstumus nuo kiekvieno tasko iki kiekvieno centroido
        distances = np.zeros((n, k))

        # Calculate distances between each sample and centroids
        #Einame per visus taskus
        for i in range(n):
            for j in range(k):
                #Pasiskaicuoja atstumus nuo kiekvieno centroido
                distances[i, j] = calculate_distance(X[i], centroids[j])

        #Taskams priskiriamu klasteriu masyvas
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

        #Atnaujiname priskyrima klasteriui
        labels = new_labels

        #Einame per centroidus ir atnaujiname visus centroidu pozicijas
        for j in range(k):
            centroids[j] = np.mean(X[labels == j], axis=0)

    return labels, centroids


def som(X, som_shape, sigma=1.0, learning_rate=0.5, max_iter=100):
    n, d = X.shape
    #Klasteriu kiekis
    som_size = som_shape[0] * som_shape[1]
    #Padaromas SOM grid
    som = np.random.randn(som_size, d)

    for _ in range(max_iter):
        #Ivesties vektorius
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

        #Gauso kaimynystes funkcija
        neighborhood = np.exp(-distance_from_winner ** 2 / (2 * sigma ** 2))

        som += learning_rate * neighborhood[:, None] * (x - som)

    som_labels = np.zeros(n)

    # Kai suformuoti klasteriai, taskai priskiriami jiems
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
feature_columns = ['Water_(g)', 'Energ_Kcal', 'Protein_(g)' ,'Lipid_Tot_(g)', 'Carbohydrt_(g)',
                    'Fiber_TD_(g)', 'Sugar_Tot_(g)', 'Iron_(mg)', 'Cholestrl_(mg)']


X = data[feature_columns].values

# Normalize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# K-means clustering
k = 5
kmeans_labels, kmeans_centroids = kmeans(X, k)
# Calculate inertia
inertia = calculate_inertia(X, kmeans_centroids, kmeans_labels)
print("Inertia:", inertia)
kmeans_silhouette = np.mean(calculate_silhouette_coefficient(X, kmeans_labels))
# SOM clustering
som_shape = (5, 1)
sigma = 1.0
learning_rate = 0.5
som_labels, som = som(X, som_shape, sigma, learning_rate)
som_silhouette = 0.0
if len(np.unique(som_labels)) > 2:
    print("Skipping - Only 1 cluster found.")
    som_silhouette = np.mean(calculate_silhouette_coefficient(X, kmeans_labels))
# Calculate inertia
inertias = []
for k_val in range(1, 10):
    kmeans_labels, kmeans_centroids = kmeans(X, k_val)
    inertia = calculate_inertia(X, kmeans_centroids, kmeans_labels)
    inertias.append(inertia)

# Calculate silhouette coefficients for different values of k
silhouette_scores = []
for k_val in range(2, 10):
    kmeans_labels, kmeans_centroids = kmeans(X, k_val)
    silhouette_score = np.mean(calculate_silhouette_coefficient(X, kmeans_labels))
    silhouette_scores.append(silhouette_score)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(X[:, 0], X[:, 1], c=kmeans_labels)
axes[0].scatter(kmeans_centroids[:, 0], kmeans_centroids[:, 1], s=100, marker='x', c='black')
axes[0].set_title(f"K-means clustering (Silhouette score: {kmeans_silhouette:.2f})")
som_silhouette = np.mean(calculate_silhouette_coefficient(X, kmeans_labels))
axes[1].scatter(X[:, 0], X[:, 1], c=som_labels)
axes[1].scatter(som[:, 0], som[:, 1], s=100, marker='x', c='black')
axes[1].set_title(f"SOM klasteriai : {som_silhouette:.2f})")
# Plot inertia values
k_values = range(1, 10)
axes[2].plot(k_values, inertias, marker='o')
axes[2].set_xlabel('Number of Clusters (k)')
axes[2].set_ylabel('Inertia')
axes[2].set_title('Inertia vs. Number of Clusters')
# Plot silhouette coefficients
k_values = range(2, 10)
axes[3].plot(k_values, silhouette_scores, marker='o')
axes[3].set_xlabel('Number of Clusters (k)')
axes[3].set_ylabel('Silhouette Coefficient')
axes[3].set_title('Silhouette Coefficient vs. Number of Clusters')

plt.tight_layout()
plt.show()