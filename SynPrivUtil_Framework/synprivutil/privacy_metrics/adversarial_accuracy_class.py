import numpy as np
import pandas as pd
from scipy.spatial import distance

from SynPrivUtil_Framework.synprivutil.privacy_metrics import PrivacyMetricCalculator


class AdversarialAccuracyCalculator(PrivacyMetricCalculator):
    def __init__(self, original_data: pd.DataFrame, synthetic_data: pd.DataFrame, distance_metric: str = 'euclidean'):
        super().__init__(original_data, synthetic_data)
        self.metric = distance_metric

    def evaluate(self):
        """
        Calculate the Nearest Neighbor Adversarial Accuracy (NNAA).

        Parameters:
        source (pd.DataFrame): Source distribution samples (n_s x d).
        target (pd.DataFrame): Target distribution samples (n_t x d).
        metric (str): Distance metric to use (default is 'euclidean').

        Returns:
        float: The calculated NNAA.
            """

        # Convert pandas DataFrames to numpy arrays
        source = self.original_data.to_numpy()
        target = self.synthetic_data.to_numpy()

        # Calculate minimum distances
        min_d_st, min_d_ts, min_d_ss, min_d_tt = self._calculate_min_distances()

        # NOTE: Based on the paper, we draw the same number of empirical samples from each dataset, so n_s = n_t
        n_s = len(source)
        n_t = len(target)

        # reference paper: https://github.com/yknot/ESANN2019/blob/master/metrics/nn_adversarial_accuracy.py
        term1 = np.sum(min_d_ts > min_d_tt) / n_t
        term2 = np.sum(min_d_st > min_d_ss) / n_s

        nnaa_value = 0.5 * (term1 + term2)

        return nnaa_value

    def _calculate_min_distances(self):
        """
        Calculate the minimum distances from each source sample to target samples
        and from each target sample to source samples using a specified distance metric.

        Parameters:
        source (pd.DataFrame): Source distribution samples (n_s x d).
        target (pd.DataFrame): Target distribution samples (n_t x d).
        metric (str): Distance metric to use (default is 'euclidean').

        Returns:
        tuple: (min_d_st, min_d_ts, min_d_ss, min_d_tt)
               - min_d_st: Minimum distance from each target sample to source samples.
               - min_d_ts: Minimum distance from each source sample to target samples.
               - min_d_ss: Minimum leave-one-out distance within source samples.
               - min_d_tt: Minimum leave-one-out distance within target samples.
        """

        # Convert pandas DataFrames to numpy arrays
        source = self.original_data.to_numpy()
        target = self.synthetic_data.to_numpy()

        # Calculate distances from target to source
        d_st = distance.cdist(target, source, metric=self.metric)
        min_d_st = np.min(d_st, axis=1)

        # Calculate distances from source to target
        d_ts = distance.cdist(source, target, metric=self.metric)
        min_d_ts = np.min(d_ts, axis=1)

        # Calculate distances within source samples (leave-one-out)
        d_ss = distance.cdist(source, source, metric=self.metric)
        np.fill_diagonal(d_ss, np.inf)  # Ignore self-distances
        min_d_ss = np.min(d_ss, axis=1)

        # Calculate distances within target samples (leave-one-out)
        d_tt = distance.cdist(target, target, metric=self.metric)
        np.fill_diagonal(d_tt, np.inf)  # Ignore self-distances
        min_d_tt = np.min(d_tt, axis=1)

        return min_d_st, min_d_ts, min_d_ss, min_d_tt