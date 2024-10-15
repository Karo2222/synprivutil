import numpy as np
from rdt import HyperTransformer
from scipy.spatial import distance
import pandas as pd

from SynPrivUtil_Framework.synprivutil.models.transform import transform_and_normalize
from SynPrivUtil_Framework.synprivutil.privacy_metrics.privacy_metric_calculator import PrivacyMetricCalculator


# NOTE: Weighted DCR


class DCRCalculator(PrivacyMetricCalculator):
    def __init__(self, original_data: pd.DataFrame, synthetic_data: pd.DataFrame, distance_metric: str = 'euclidean', weights: np.ndarray = None):
        """
        Initializes the DCRCalculator with original and synthetic datasets and a specified distance metric.

        Parameters:
        original_data: The original dataset as a pandas DataFrame.
        synthetic_data: The synthetic dataset as a pandas DataFrame.
        distance_metric: The distance metric to use ('euclidean', 'cityblock', 'hamming', etc.).
        """
        super().__init__(original_data, synthetic_data, distance_metric=distance_metric)
        if distance_metric is None:
            raise ValueError("Parameter 'distance_metric' is required in DCRCalculator.")
        self.distance_metric = distance_metric
        self.weights = weights if weights is not None else np.ones(original_data.shape[1])

    def evaluate(self) -> float:
            """
            Calculates the Distance of Closest Record (DCR) between the original and synthetic datasets.

            :return: The average DCR value.
            """            # Apply weights to the original and synthetic data
            weighted_original_data = self.original_data * self.weights
            weighted_synthetic_data = self.synthetic_data * self.weights
            # Calculate distances using cdist
            dists = distance.cdist(weighted_synthetic_data, weighted_original_data, metric=self.distance_metric)

            # Find the minimum distance for each synthetic record
            min_distances = np.min(dists, axis=1)
            #print(f"min dist {min_distances}")

            # Return the average DCR
            return np.mean(min_distances)

    def set_metric(self, metric: str):
        """
        Updates the distance metric used in DCR calculation.

        :param metric: The new distance metric to use.
        """
        self.distance_metric = metric

# # Test DataFrames
# original_data = pd.DataFrame({
#     'A': [1, 2, 3],
#     'B': [1, 2, 3]
# })
#
# synthetic_data = pd.DataFrame({
#     'A': [1, 3, 5],
#     'B': [1, 3, 5]
# })
#
# # Equal weights
# weights = np.array([0.5, 0.5])
#
# arr = np.array([[1, 2], [3, 4], [5, 6]])
# print(arr.shape)  # Output: (3, 2) - 3 rows and 2 columns
#
# df = pd.DataFrame({
#     'A': [1, 2, 3],
#     'B': [4, 5, 6]
# })
#
# print(df.shape)  # Output: (3, 2) - 3 rows and 2 columns
#
# # Calculate weighted DCR (with equal weights)
# weighted_dcr_calculator = DCRCalculator(original_data, synthetic_data, weights=weights)
# weighted_dcr = weighted_dcr_calculator.evaluate()
#
# # Calculate unweighted DCR
# dcr_calculator = DCRCalculator(original_data, synthetic_data)
# unweighted_dcr = dcr_calculator.evaluate()
# print(f"Weighted {weighted_dcr}")
# print(f"Un-Weighted {unweighted_dcr}")
# # Assert they are the same
# #assert np.isclose(unweighted_dcr, weighted_dcr), "Equal weights should yield the same DCR."
#
#

if __name__ == "__main__":
    synthetic_datasets = ["copulagan", "ctgan", "gaussian_copula", "gmm", "tvae", "random"]
    original_datasets =["diabetes", "cardio", "insurance"]

    for orig in original_datasets:
        for syn in synthetic_datasets:
            original_data = pd.read_csv(f"/Users/ksi/Development/Bachelorthesis/{orig}.csv")
            synthetic_data = pd.read_csv(f"/Users/ksi/Development/Bachelorthesis/SynPrivUtil_Framework/synprivutil/models/{orig}_datasets/{syn}_sample.csv")
            transformed_data_o, transformed_data_s = transform_and_normalize(original_data, synthetic_data)
            test_dcr_calculator = DCRCalculator(transformed_data_o, transformed_data_s)
            test_dcr = test_dcr_calculator.evaluate()
            print(f'{test_dcr}')

