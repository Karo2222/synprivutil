import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from SynPrivUtil_Framework.synprivutil.models.transform import transform_and_normalize
from SynPrivUtil_Framework.synprivutil.privacy_metrics import PrivacyMetricCalculator, PrivacyMetricManager


class AdversarialAccuracyCalculator(PrivacyMetricCalculator):
    def __init__(self, original_data: pd.DataFrame, synthetic_data: pd.DataFrame, distance_metric: str = 'euclidean'):
        super().__init__(original_data, synthetic_data)
        self.distance_metric = distance_metric

    def evaluate(self):
        """
        Calculate the Nearest Neighbor Adversarial Accuracy (NNAA).

        Parameters:
        original (pd.DataFrame): Source distribution samples (n_s x d).
        synthetic (pd.DataFrame): Target distribution samples (n_t x d).
        metric (str): Distance metric to use (default is 'euclidean').

        Returns:
        float: The calculated NNAA.
            """

        # Calculate minimum distances
        min_d_syn_orig, min_d_orig_syn, min_d_orig_orig, min_d_syn_syn = self._calculate_min_distances()

        # reference paper: https://github.com/yknot/ESANN2019/blob/master/metrics/nn_adversarial_accuracy.py
        term1 = np.mean(min_d_orig_syn > min_d_orig_orig)
        term2 = np.mean(min_d_syn_orig > min_d_syn_syn)

        nnaa_value = 0.5 * (term1 + term2)

        return nnaa_value

    def _calculate_min_distances(self):
        """
        Calculate the minimum distances from each original sample to synthetic samples
        and from each synthetic sample to original samples using a specified distance metric.

        Parameters:
        original (pd.DataFrame): Source distribution samples (n_s x d).
        synthetic (pd.DataFrame): Target distribution samples (n_t x d).
        metric (str): Distance metric to use (default is 'euclidean').

        Returns:
        tuple: (min_d_syn_orig, min_d_orig_syn, min_d_orig_orig, min_d_syn_syn)
               - min_d_syn_orig: Minimum distance from each synthetic sample to original samples.
               - min_d_orig_syn: Minimum distance from each original sample to synthetic samples.
               - min_d_orig_orig: Minimum leave-one-out distance within original samples.
               - min_d_syn_syn: Minimum leave-one-out distance within synthetic samples.
        """

        # Calculate distances from synthetic to original
        d_syn_orig = distance.cdist(self.synthetic_data, self.original_data, metric=self.distance_metric)
        min_d_syn_orig = np.min(d_syn_orig, axis=1)

        # Calculate distances from original to synthetic
        d_orig_syn = distance.cdist(self.original_data, self.synthetic_data, metric=self.distance_metric)
        min_d_orig_syn = np.min(d_orig_syn, axis=1)

        # Calculate distances within original samples (leave-one-out)
        d_orig_orig = distance.cdist(self.original_data, self.original_data, metric=self.distance_metric)
        np.fill_diagonal(d_orig_orig, np.inf)  # Ignore self-distances
        min_d_orig_orig = np.min(d_orig_orig, axis=1)

        # Calculate distances within synthetic samples (leave-one-out)
        d_syn_syn = distance.cdist(self.synthetic_data, self.synthetic_data, metric=self.distance_metric)
        np.fill_diagonal(d_syn_syn, np.inf)  # Ignore self-distances
        min_d_syn_syn = np.min(d_syn_syn, axis=1)

        return min_d_syn_orig, min_d_orig_syn, min_d_orig_orig, min_d_syn_syn


class NearestNeighborMetrics:
    """Calculate nearest neighbors and metrics for original and synthetic data."""

    def __init__(self, original_data, synthetic_data):
        """Initialize with original and synthetic data."""
        self.data = {'original': original_data, 'synthetic': synthetic_data}
        self.dists = {}

    def nearest_neighbors(self, t, s):
        """Find nearest neighbors between two datasets (t and s)."""
        nn_s = NearestNeighbors(n_neighbors=1).fit(self.data[s])
        if t == s:
            # Find distances within the same dataset
            d = nn_s.kneighbors()[0]
        else:
            # Find distances between different datasets
            d = nn_s.kneighbors(self.data[t])[0]

        return t, s, d

    def compute_nn(self):
        """Compute nearest neighbors for original and synthetic datasets."""
        pairs = [('original', 'original'), ('original', 'synthetic'), ('synthetic', 'synthetic'), ('synthetic', 'original')]
        for (t, s) in tqdm(pairs):
            t, s, d = self.nearest_neighbors(t, s)
            self.dists[(t, s)] = d

    def adversarial_accuracy(self):
        """Calculate the adversarial accuracy score between original and synthetic data."""
        orig_vs_synth = np.mean(self.dists[('original', 'synthetic')] > self.dists[('original', 'original')])
        synth_vs_orig = np.mean(self.dists[('synthetic', 'original')] > self.dists[('synthetic', 'synthetic')])
        return 0.5 * (orig_vs_synth + synth_vs_orig)


if __name__ == "__main__":
    synthetic_datasets = ["copulagan", "ctgan", "gaussian_copula", "gmm", "tvae", "random"]
    original_datasets =["diabetes", "cardio", "insurance"]

    for orig in original_datasets:
        for syn in synthetic_datasets:
            original_data = pd.read_csv(f"/Users/ksi/Development/Bachelorthesis/{orig}.csv")
            synthetic_data = pd.read_csv(f"/Users/ksi/Development/Bachelorthesis/SynPrivUtil_Framework/synprivutil/models/{orig}_datasets/{syn}_sample.csv")
            transformed_data_o, transformed_data_s = transform_and_normalize(original_data, synthetic_data)
            # print(f'~~~~~~Adversarial Accuracy WITH NEAREST NEIGHBOR~~~~~~')
            nnm = NearestNeighborMetrics(transformed_data_o, transformed_data_s)

            # Compute nearest neighbor distances
            nnm.compute_nn()

            # Calculate adversarial accuracy
            adversarial_accuracy = nnm.adversarial_accuracy()


            # Output adversarial accuracy score
            print(f'{adversarial_accuracy}')
            #
            # print(f'~~~~~~Adversarial Accuracy CDIST~~~~~~')
            #
            # calc = AdversarialAccuracyCalculator(transformed_data_o, transformed_data_s)
            # nnaa = calc.evaluate()
            # print(nnaa)