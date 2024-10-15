import numpy as np
import pandas as pd
from rdt import HyperTransformer
from scipy.spatial import distance

from SynPrivUtil_Framework.synprivutil.models.transform import transform_and_normalize
from SynPrivUtil_Framework.synprivutil.privacy_metrics import PrivacyMetricCalculator


class NNDRCalculator(PrivacyMetricCalculator):
    def __init__(self, original_data: pd.DataFrame, synthetic_data: pd.DataFrame, distance_metric: str = 'euclidean'):
        """
        Initialize the NNDRCalculator with the original and synthetic datasets.

        Parameters:
        - original_data: pd.DataFrame
          The original dataset.
        - synthetic_data: pd.DataFrame
          The synthetic dataset generated from the original data.
        - metric: str (default: 'euclidean')
          The distance metric to use for calculating NNDR.
        """
        super().__init__(original_data, synthetic_data, distance_metric=distance_metric)
        if distance_metric is None:
            raise ValueError("Parameter 'metric' is required in NNDRCalculator.")
        self.distance_metric = distance_metric

    def evaluate(self) -> float:
        """
        Calculate the Nearest Neighbor Distance Ratio (NNDR) for the synthetic data.

        Returns:
        - nndr: The mean NNDR value for the synthetic dataset.
        """

        #print("BEGIN DISTANCES")
        # Compute distances from each synthetic record to all original records
        distances = distance.cdist(self.synthetic_data, self.original_data, metric=self.distance_metric)

        # # Sort distances for each synthetic record
        # sorted_distances = np.sort(distances, axis=1)
        #
        # # Get the nearest and second nearest distances
        # nearest_distances = sorted_distances[:, 0]
        # second_nearest_distances = sorted_distances[:, 1]
        #print(f"DISTANCES {distances}")
        # Use np.partition to find the smallest and second smallest distances
        partitioned_distances = np.partition(distances, 1, axis=1)[:, :2]
        # print("PARTITIONED")
        nearest_distances = partitioned_distances[:, 0]
        second_nearest_distances = partitioned_distances[:, 1]
        # Calculate NNDR for each synthetic record
        nndr_list = nearest_distances / (second_nearest_distances+1e-16)  # to prevent division by 0

        # Calculate the mean NNDR
        nndr = np.mean(nndr_list)
        return nndr


if __name__ == "__main__":
    #"copulagan", "ctgan", "gaussian_copula", "gmm", "tvae",
    #"diabetes", "cardio",
    synthetic_datasets = ["copulagan", "ctgan", "gaussian_copula", "gmm", "tvae", "random"]
    original_datasets =["diabetes", "cardio", "insurance"]

    for orig in original_datasets:
        for syn in synthetic_datasets:
            original_data = pd.read_csv(f"/Users/ksi/Development/Bachelorthesis/{orig}.csv")
            synthetic_data = pd.read_csv(f"/Users/ksi/Development/Bachelorthesis/SynPrivUtil_Framework/synprivutil/models/{orig}_datasets/{syn}_sample.csv")
            transformed_data_o, transformed_data_s = transform_and_normalize(original_data, synthetic_data)
            test_nndr_calculator = NNDRCalculator(transformed_data_o, transformed_data_s)
            test_nndr = test_nndr_calculator.evaluate()
            print(f'{test_nndr}')