import numpy as np
from sklearn.cluster import DBSCAN

def group_by_dbscan(values, eps=0.5, min_samples=2):
    """
    Groups floating-point values by proximity using DBSCAN.

    :param values: list or array of floats
    :param eps: maximum distance between two samples for one to be considered
                as in the neighborhood of the other
    :param min_samples: the minimum number of samples in a neighborhood for
                        a point to be considered a core point
    :return: list of clusters, each cluster is a list of floats
    """
    # Convert to 2D array for DBSCAN (expects [ [x1], [x2], ... ])
    data = np.array(values).reshape(-1, 1)
    
    # Run DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
    
    # Extract labels (-1 label means "noise" point)
    labels = db.labels_
    
    # Organize points by cluster
    clusters_dict = {}
    for label, val in zip(labels, values):
        if label not in clusters_dict:
            clusters_dict[label] = []
        clusters_dict[label].append(val)
    
    # Sort out noise label (-1) if you want to exclude it or treat it separately
    clusters = []
    for label, cluster_vals in clusters_dict.items():
        if label == -1:
            # Noise points
            continue
        clusters.append(cluster_vals)
    
    return clusters