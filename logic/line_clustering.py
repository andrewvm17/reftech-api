# logic/clustering.py

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def cluster_lines(lines, eps=0.5, min_samples=2):
    """
    Cluster line segments given as [x1, y1, x2, y2, slope] based on their (slope, intercept).

    Returns:
        dict: A dictionary mapping cluster labels to lists of lines.
    """
    features = []
    for line in lines:
        x1, y1, x2, y2, m = line
        # Compute y-intercept b = y1 - m * x1
        b = y1 - m * x1
        features.append([m, b])
    features = np.array(features)

    # Standardize features to balance scale differences between slope and intercept
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Cluster using DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(features_scaled)
    labels = clustering.labels_
    
    clusters = {}
    for label, line in zip(labels, lines):
        clusters.setdefault(label, []).append(line)
    
    return clusters

def best_fit_line_for_cluster(cluster_lines):
    """
    Given a list of line segments (each as [x1, y1, x2, y2, slope]),
    compute a best-fit line (using linear regression on all endpoints).
    
    Returns:
        list: A representative line in the form [x_start, y_start, x_end, y_end, slope].
              In the nearly vertical case, slope is returned as None.
    """
    points = []
    for line in cluster_lines:
        x1, y1, x2, y2, _ = line
        points.append([x1, y1])
        points.append([x2, y2])
    points = np.array(points, dtype=np.float64)
    
    # If the x-variance is near zero, handle as vertical line.
    if np.std(points[:, 0]) < 1e-6:
        x_val = np.mean(points[:, 0])
        y_min = np.min(points[:, 1])
        y_max = np.max(points[:, 1])
        return [x_val, y_min, x_val, y_max, None]
    else:
        # Compute linear regression (best-fit line) using np.polyfit.
        m, b = np.polyfit(points[:, 0], points[:, 1], 1)
        x_min = np.min(points[:, 0])
        x_max = np.max(points[:, 0])
        y_min = m * x_min + b
        y_max = m * x_max + b
        return [x_min, y_min, x_max, y_max, m]

def best_fit_lines_from_clusters(clusters, ignore_noise=True):
    """
    For each cluster, compute the best-fitting line.

    Args:
        clusters (dict): Mapping from cluster label to list of lines.
        ignore_noise (bool): If True, skip cluster with label -1 (noise).

    Returns:
        dict: Mapping from cluster label to best-fit line.
    """
    best_fit = {}
    for label, lines in clusters.items():
        if ignore_noise and label == -1:
            continue
        best_fit[label] = best_fit_line_for_cluster(lines)
    return best_fit
