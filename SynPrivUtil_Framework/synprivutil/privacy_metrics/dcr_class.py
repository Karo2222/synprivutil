import numpy as np
from scipy.spatial import distance
import pandas as pd

from SynPrivUtil_Framework.synprivutil.privacy_metrics.privacy_metric_calculator import PrivacyMetricCalculator


# NOTE: Weighted DCR


class DCRCalculator(PrivacyMetricCalculator):
    def __init__(self, original_data: pd.DataFrame, synthetic_data: pd.DataFrame, distance_metric: str = 'euclidean'):
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
        self.metric = distance_metric

    def evaluate(self) -> float:
        """
        Calculates the Distance of Closest Record (DCR) between the original and synthetic datasets.

        :return: The average DCR value.
        """
        dcr_values = []

        for _, syn_record in self.synthetic_data.iterrows():
            min_distance = self._calculate_min_distance(syn_record)
            dcr_values.append(min_distance)

        avg_dcr = np.mean(dcr_values)
        return avg_dcr

    def _calculate_min_distance(self, syn_record: pd.Series) -> float:
        """
        Calculates the minimum distance between a synthetic record and all records in the original dataset.

        :param syn_record: A single record from the synthetic dataset.
        :return: The minimum distance value.
        """

        # Compute distances from the synthetic record to all original records
        distances = distance.cdist([syn_record], self.original_data, metric=self.metric)
        min_distance = np.min(distances)
        return min_distance

    def set_metric(self, metric: str):
        """
        Updates the distance metric used in DCR calculation.

        :param metric: The new distance metric to use.
        """
        self.metric = metric
