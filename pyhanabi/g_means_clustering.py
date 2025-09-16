import torch
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from scipy.stats import anderson

def is_gaussian(data, significance_level=0.05):
    """
    Test if the data distribution is Gaussian using Anderson-Darling test.
    Returns True if Gaussian, False otherwise.
    """
    if data.shape[0] < 10:  # too few samples to test reliably
        return True
    stat, _, critical_values = anderson(data, dist='norm')
    # Compare statistic with critical value for ~5% significance
    return stat < critical_values[2]  # index 2 ≈ 5% level

def gmeans_clustering(X, seed=42, max_k=50):

    """
    Perform G-means style adaptive clustering.
    X: numpy array of shape (n_samples, n_features)
    """
    clusters = [X]
    final_labels = np.zeros(X.shape[0], dtype=int)
    kmeans_models = []
    current_k = 1
    converged = False

    while not converged:
        converged = True
        new_clusters = []
        new_labels = final_labels.copy()
        cluster_id_offset = 0

        for cluster_id, cluster_data in enumerate(clusters):
            if cluster_data.shape[0] <= 1:
                continue
            
            # Run kmeans with k=2 on this cluster
            kmeans = KMeans(n_clusters=2, random_state=seed, n_init=10)
            sub_labels = kmeans.fit_predict(cluster_data)
            
            # Project onto vector between centroids
            centroids = kmeans.cluster_centers_
            direction = centroids[1] - centroids[0]
            direction /= np.linalg.norm(direction)
            projected = np.dot(cluster_data, direction)

            # Test Gaussianity
            if is_gaussian(projected):
                # Keep as one cluster
                new_clusters.append(cluster_data)
                kmeans1 = KMeans(n_clusters=1, random_state=seed, n_init=10).fit(cluster_data)
                kmeans_models.append(kmeans1)
                new_labels[np.all(X[:, None] == cluster_data[None, :], axis=2).any(axis=1)] = cluster_id_offset
                cluster_id_offset += 1
            else:
                # Split into two clusters
                converged = False
                for sub_id in [0, 1]:
                    sub_cluster = cluster_data[sub_labels == sub_id]
                    if sub_cluster.shape[0] > 0:
                        new_clusters.append(sub_cluster)
                        new_labels[np.all(X[:, None] == sub_cluster[None, :], axis=2).any(axis=1)] = cluster_id_offset
                        cluster_id_offset += 1

        clusters = new_clusters
        final_labels = new_labels
        current_k = cluster_id_offset
        if current_k >= max_k:
            break

    return final_labels, current_k