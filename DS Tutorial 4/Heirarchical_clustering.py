import numpy as np
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

# Our complex linkage function
def complex_linkage(cluster1, cluster2):
    # All pairwise distances
    dists = [euclidean(p1, p2) for p1 in cluster1 for p2 in cluster2]
    min_dist = min(dists)
    max_dist = max(dists)

    # Centroid distance
    centroid1 = np.mean(cluster1, axis=0)
    centroid2 = np.mean(cluster2, axis=0)
    centroid_dist = euclidean(centroid1, centroid2)

    # Weighted combination
    return 0.4 * min_dist + 0.4 * max_dist + 0.2 * centroid_dist

# Agglomerative clustering using the above linkage
def hierarchical_clustering(X, num_clusters=2):
    clusters = [[x] for x in X]  # Start with each point as its own cluster

    while len(clusters) > num_clusters:
        min_dist = float('inf')
        to_merge = (None, None)

        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                dist = complex_linkage(clusters[i], clusters[j])
                if dist < min_dist:
                    min_dist = dist
                    to_merge = (i, j)

        i, j = to_merge
        # Merge clusters i and j
        merged = clusters[i] + clusters[j]
        clusters.pop(j)  # pop the later index first
        clusters.pop(i)
        clusters.append(merged)

    return clusters

# Sample data
from sklearn.datasets import make_blobs
X, _ = make_blobs(n_samples=15, centers=3, random_state=42)

# Run clustering
clusters = hierarchical_clustering(X, num_clusters=3)

# Plot
colors = ['r', 'g', 'b', 'c', 'm']
for i, cluster in enumerate(clusters):
    cluster = np.array(cluster)
    plt.scatter(cluster[:, 0], cluster[:, 1], color=colors[i % len(colors)], label=f"Cluster {i+1}")
plt.title("Hierarchical Clustering with Complex Linkage")
plt.legend()
plt.grid(True)
plt.show()
