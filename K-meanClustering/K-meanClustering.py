from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

def dist(p, q):
    return np.sqrt(np.sum((p - q) ** 2))

def normalize(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / std

def init(k, X):
    centroid = {}
    for i in range(k):
        # Choose random points from the dataset as initial centroids
        random_idx = np.random.randint(0, X.shape[0])
        centroid[i] = {
            'center': X[random_idx],
            'colour': colours[i],
            'points': []
        }
    return centroid

def assignPtsToCluster(X, centroid):
    for i in range(len(centroid)):
        centroid[i]['points'] = []  # Reset points for each cluster

    m = X.shape[0]
    for i in range(m):
        cdist = []
        cX = X[i]
        for kC in range(k):
            d = dist(centroid[kC]['center'], cX)
            cdist.append(d)

        cluster_id = np.argmin(cdist)
        centroid[cluster_id]['points'].append(cX)

    return centroid

def updateClusters(centroid):
    for i in range(k):
        pts = np.array(centroid[i]['points'])

        if pts.shape[0] > 0:
            newCenter = pts.mean(axis=0)
            centroid[i]['center'] = newCenter
    return centroid

def plotClusters(centroid, X, title=""):
    plt.figure(figsize=(8, 6))
    for kX in range(k):
        pts = np.array(centroid[kX]['points'])

        if pts.shape[0] > 0:
            plt.scatter(pts[:, 0], pts[:, 1], color=centroid[kX]['colour'], label=f'Cluster {kX + 1}')
            uk = centroid[kX]['center']
            plt.scatter(uk[0], uk[1], color='black', marker='*', s=150, label=f'Centroid {kX + 1}')
    plt.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.2, label="Unclustered Points")
    plt.title(title)
    # plt.legend()
    plt.show()

# Generate data
X, y = make_blobs(n_samples=500, n_features=2, centers=5, random_state=42)
X = normalize(X)

# Parameters
k = 5
n_features = X.shape[1]
colours = ["red", "green", "blue", "yellow", "orange"]

# K-means algorithm
centroid = init(k, X)
max_iterations = 100
tolerance = 1e-4

for iteration in range(max_iterations):
    old_centroids = {i: centroid[i]['center'].copy() for i in range(k)}
    centroid = assignPtsToCluster(X, centroid)
    centroid = updateClusters(centroid)

    # Convergence check
    converged = True
    for i in range(k):
        if dist(old_centroids[i], centroid[i]['center']) > tolerance:
            converged = False
            break

    # Plot clusters after each iteration
    # plotClusters(centroid, X, title=f"Iteration {iteration + 1}")

    if converged:
        print(f"Converged after {iteration + 1} iterations")
        break
else:
    print("Reached maximum iterations without convergence")

# Final Plot
plotClusters(centroid, X, title="Final Clusters")
