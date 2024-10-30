import numpy as np
from scipy.spatial import distance

def calculate_nndr(real_data, synthetic_data, metric='euclidean'):
    """
    Calculate the Nearest Neighbor Distance Ratio (NNDR) for synthetic data.
    
    Parameters:
    real_data (pd.DataFrame): The original dataset.
    synthetic_data (pd.DataFrame): The synthetic dataset.
    metric (str): The distance metric to use ('euclidean', 'manhattan', 'hamming', etc.).
    
    Returns:
    float: The average NNDR value.
    """
    # Calculate pairwise distances between synthetic and real datasets using cdist
    distances = distance.cdist(synthetic_data, real_data, metric=metric)
    
    # Calculate NNDR for each synthetic record
    nearest_distances = []
    for dist in distances:
        sorted_distances = np.sort(dist)
        nearest_distance = sorted_distances[0]
        second_nearest_distance = sorted_distances[1]
        nearest_distances.append(nearest_distance / second_nearest_distance)
    
    # Calculate the average NNDR
    nndr = np.mean(nearest_distances)
    return nndr