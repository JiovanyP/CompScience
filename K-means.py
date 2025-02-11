def euclidean_distance(p1, p2):
    """Calculate the Euclidean distance between two points."""
    return sum((p1[i] - p2[i]) ** 2 for i in range(len(p1))) ** 0.5

def initialize_centroids(data, k):
    """Manually select the first `k` unique points as initial centroids."""
    centroids = []
    step = max(1, len(data) // k)  
    for i in range(k):
        centroids.append(data[i * step])
    return centroids

def assign_clusters(data, centroids):
    """Assign each data point to the closest centroid."""
    clusters = [[] for _ in centroids]
    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        closest_idx = distances.index(min(distances))
        clusters[closest_idx].append(point)
    return clusters

def compute_new_centroids(clusters):
    """Calculate new centroids as the mean of all points in each cluster."""
    new_centroids = []
    for cluster in clusters:
        if cluster:
            centroid = [sum(dim) / len(cluster) for dim in zip(*cluster)]
        else:
            centroid = [0, 0] 
        new_centroids.append(centroid)
    return new_centroids

def k_means(data, k, max_iters=100):
    """Main K-Means algorithm."""
    centroids = initialize_centroids(data, k)
    
    for _ in range(max_iters):
        clusters = assign_clusters(data, centroids)
        new_centroids = compute_new_centroids(clusters)

        if new_centroids == centroids: 
            break
        
        centroids = new_centroids
    
    return clusters, centroids


data = [
    [1, 2], [2, 3], [3, 3], [6, 8], [7, 9], [8, 7], [12, 13], [13, 14], [14, 15]
]

# Run K-Means
k = 3
clusters, centroids = k_means(data, k)

# Output results
print("Final Cluster Centroids:", centroids)
for i, cluster in enumerate(clusters):
    print(f"Cluster {i + 1}: {cluster}")
