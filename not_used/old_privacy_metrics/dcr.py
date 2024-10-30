import numpy as np
from scipy.spatial import distance
import pandas as pd

from privacy_utility_framework.privacy_utility_framework.privacy_metrics import DCRCalculator

# Example Usage:
if __name__ == "__main__":
    # Example data loading
    original_df = pd.read_csv('path_to_original.csv')
    synthetic_df = pd.read_csv('path_to_synthetic.csv')

    # Initialize DCR calculator
    dcr_calculator = DCRCalculator(original=original_df, synthetic=synthetic_df, distance_metric='euclidean')

    # Validate datasets
    dcr_calculator.validate_datasets()

    # Calculate DCR
    avg_dcr = dcr_calculator.calculate_dcr()
    print(f"Average DCR (Euclidean): {avg_dcr}")

    # Example: Switching to Manhattan distance
    dcr_calculator.set_metric('cityblock')
    avg_dcr_manhattan = dcr_calculator.calculate_dcr()
    print(f"Average DCR (Manhattan): {avg_dcr_manhattan}")

def calculate_dcr(original_data, synthetic_data, metric='euclidean'):
    """
    Calculate the Distance to Closest Record (DCR) for synthetic data.
    
    Args:
    original_data (np.array): Original dataset as a 2D NumPy array.
    synthetic_data (np.array): Synthetic dataset as a 2D NumPy array.
    metric (str): The distance metric to use ('euclidean', 'hamming', 'cityblock').

    Euclidean Distance: Measures the straight-line distance between two points in Euclidean space.
    Hamming Distance: Measures the number of positions at which the corresponding elements differ, suitable for binary vectors.
    Manhattan Distance: Measures the sum of the absolute differences between the coordinates of a pair of points.
    
    Returns:
    float: Average DCR for the synthetic data.
    """
    

    
    # Initialize a list to store the DCR for each synthetic record
    dcr_list = []
    
    # Calculate DCR for each synthetic record
    for syn_record in synthetic_data:
        # Compute distances from the synthetic record to all original records
        distances = distance.cdist([syn_record], original_data, metric=metric)
        # Find the minimum distance (DCR)
        min_distance = np.min(distances)
        dcr_list.append(min_distance)
    
    # Calculate the average DCR
    average_dcr = np.mean(dcr_list)
    
    return average_dcr

if __name__ == "__main__":
    # Example
    original_data = np.array([[1, 2], [3, 4], [5, 6]])
    synthetic_data = np.array([[1.1, 2.1], [3.1, 4.1], [5.1, 6.1]])

    # Calculate DCR using Euclidean distance
    dcr_euclidean = calculate_dcr(original_data, synthetic_data, metric='euclidean')
    print(f"Average DCR (Euclidean): {dcr_euclidean}")

    # Calculate DCR using Hamming distance
    original_data_binary = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 0]])
    synthetic_data_binary = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]])

    dcr_hamming = calculate_dcr(original_data_binary, synthetic_data_binary, metric='hamming')
    print(f"Average DCR (Hamming): {dcr_hamming}")

    # Calculate DCR using Manhattan (Cityblock) distance
    dcr_manhattan = calculate_dcr(original_data, synthetic_data, metric='cityblock')
    print(f"Average DCR (Manhattan): {dcr_manhattan}")
