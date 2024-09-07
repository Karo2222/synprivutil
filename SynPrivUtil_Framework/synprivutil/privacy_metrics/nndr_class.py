import numpy as np
import pandas as pd
from scipy.spatial import distance

from SynPrivUtil_Framework.synprivutil.privacy_metrics import PrivacyMetricCalculator


class NNDRCalculator(PrivacyMetricCalculator):
    # TODO: change name to dismetric
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
        super().__init__(original_data, synthetic_data, metric=distance_metric)
        if distance_metric is None:
            raise ValueError("Parameter 'metric' is required in NNDRCalculator.")
        self.metric = distance_metric

    def evaluate(self) -> float:
        """
        Calculate the Nearest Neighbor Distance Ratio (NNDR) for the synthetic data.

        Returns:
        - nndr_list: List of NNDR values for each record in the synthetic dataset.
        """
        nndr_list = []

        for _, syn_record in self.synthetic_data.iterrows():
            # Compute distances from the synthetic record to all original records
            distances = distance.cdist([syn_record], self.original_data, metric=self.metric)

            # Sort distances and get the nearest and second nearest distances
            distances = np.sort(distances[0])
            nearest_distance = distances[0]
            second_nearest_distance = distances[1] if len(distances) > 1 else nearest_distance

            # Calculate NNDR for this synthetic record
            nndr = nearest_distance / second_nearest_distance
            nndr_list.append(nndr)

        nndr = np.mean(nndr_list)
        return nndr
