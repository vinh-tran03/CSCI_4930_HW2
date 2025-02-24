import numpy as np
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt

# Compute Euclidean distance
def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

# Randomly initialize centroids
def initialize_centroids(data, k):
    return data[np.random.choice(data.shape[0], k, replace=False)]

# Assign each point to the nearest centroid
def assign_clusters(data, centroids):
    distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)  # Vectorized distance calculation
    return np.argmin(distances, axis=1)  # Assign cluster based on min distance

# Compute new centroids
def update_centroids(data, clusters, k):
    centroids = np.zeros((k, data.shape[1]))
    for i in range(k):
        cluster_points = data[np.where(clusters == i)]
        if cluster_points.shape[0] == 0:  # Handle empty clusters
            centroids[i] = data[np.random.randint(data.shape[0])]
        else:
            centroids[i] = cluster_points.mean(axis=0)
    return centroids

# Compute Sum of Squared Errors (SSE)
def compute_sse(data, clusters, centroids):
    sse = 0
    for i in range(len(centroids)):  # Loop over clusters
        cluster_points = data[clusters == i]  # Get all points in cluster i
        for point in cluster_points:  
            sse += euclidean_distance(point, centroids[i]) ** 2  # Compute squared distance
    return sse

# Compute Rand Index (RI)
def compute_rand_index(true_labels, predicted_labels):
    tp, tn, fp, fn = 0, 0, 0, 0
    for i, j in combinations(range(len(true_labels)), 2):
        same_true = true_labels[i] == true_labels[j]
        same_pred = predicted_labels[i] == predicted_labels[j]
        if same_true and same_pred:
            tp += 1
        elif not same_true and not same_pred:
            tn += 1
        elif not same_true and same_pred:
            fp += 1
        else:
            fn += 1
    return (tp + tn) / (tp + tn + fp + fn)

# K-means algorithm
def kmeans(data, k, max_iters=100):
    centroids = initialize_centroids(data, k)
    for _ in range(max_iters):
        clusters = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, clusters, k)
        if np.allclose(centroids, new_centroids):  # Use np.allclose for floating-point stability
            break
        centroids = new_centroids
    return clusters, centroids

# Run K-means multiple times and find the best SSE & RI
def run_experiment(data, true_labels, k=3, runs=10):
    best_sse, best_ri = float('inf'), 0
    best_clusters, best_centroids = None, None
    for _ in range(runs):
        clusters, centroids = kmeans(data, k)
        sse = compute_sse(data, clusters, centroids)
        ri = compute_rand_index(true_labels, clusters)
        if sse < best_sse:
            best_sse, best_clusters, best_centroids = sse, clusters, centroids
        best_ri = max(best_ri, ri)
    return best_sse, best_ri, best_clusters, best_centroids


# Load dataset
df = pd.read_csv("spiral-dataset.csv", delimiter="\t", header=None, names=["X", "Y", "Cluster"])
data = df.iloc[:, :2].values  # Extract features
true_labels = df["Cluster"].values  # Extract true cluster labels

def plot_clusters(data, clusters, centroids, title="K-Means Clustering Result"):
    k = len(np.unique(clusters))  # Number of clusters
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # Define colors for clusters
    plt.figure(figsize=(8, 6))

    # Plot each cluster with a unique color
    for i in range(k):
        cluster_points = data[np.where(clusters == i)]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                    color=colors[i % len(colors)], label=f'Cluster {i}', alpha=0.6)

    # Plot centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], 
                color='black', marker='X', s=200, label='Centroids')

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(title)
    plt.legend()
    plt.show()

# Run K-means 10 times and get best SSE & RI
best_sse, best_ri, best_clusters, best_centroids = run_experiment(data, true_labels)
print("=====================================")
print(f"Best SSE: {best_sse}")
print(f"Best Rand Index: {best_ri}")
print("=====================================")
plot_clusters(data, best_clusters, best_centroids)
