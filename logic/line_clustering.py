import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def cluster_lines(lines, eps=0.5, min_samples=2):
    """
    Cluster line segments given as [x1, y1, x2, y2, slope] based on their orientation and position.

    Parameters:
        lines (list of list of float): Each inner list is [x1, y1, x2, y2, slope].
        eps (float): The maximum distance between two samples for one to be considered as in the neighborhood
                     of the other (DBSCAN parameter). This is applied on the standardized features.
        min_samples (int): The number of samples (or total weight) in a neighborhood for a point to be considered
                           as a core point (DBSCAN parameter).

    Returns:
        dict: A dictionary where keys are cluster labels and values are lists of line segments belonging to that cluster.
              A label of -1 indicates noise.
    """
    # Create the feature vector for each line: (slope, intercept)
    features = []
    for line in lines:
        x1, y1, x2, y2, m = line
        # Compute the y-intercept b using one of the endpoints
        b = y1 - m * x1
        features.append([m, b])
    
    features = np.array(features)
    
    # Standardize features so that differences in slope and intercept are on the same scale
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Apply DBSCAN clustering on the scaled features
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(features_scaled)
    labels = clustering.labels_
    
    # Group the original lines by their cluster labels
    clusters = {}
    for label, line in zip(labels, lines):
        clusters.setdefault(label, []).append(line)
    
    return clusters

# Example usage:
if __name__ == '__main__':
    # Example list of detected lines (you can replace this with your actual detections)
    detected_lines = [
        [100, 50, 200, 80, 0.3],
        [102, 52, 198, 78, 0.31],
        [300, 400, 350, 420, 0.1],
        [305, 405, 355, 425, 0.09],
        [400, 100, 500, 150, 0.5],
        [402, 102, 498, 148, 0.49]
        # ... add as many as you have
    ]
    
    clusters = cluster_lines(detected_lines, eps=0.5, min_samples=2)
    for cluster_label, lines in clusters.items():
        print(f"Cluster {cluster_label}:")
        for l in lines:
            print("   ", l)
