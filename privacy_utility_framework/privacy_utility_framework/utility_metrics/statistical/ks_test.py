import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import ks_2samp

from privacy_utility_framework.privacy_utility_framework.models.transform import transform_and_normalize


#KSCalculator
def ks_test_columns(original, synthetic):
    """
    Calculate the Kolmogorov-Smirnov statistic for each feature.

    Parameters:
        original (pd.DataFrame): Original data.
        synthetic (pd.DataFrame): Synthetic data.

    Returns:
        dict: KS statistics for each feature.
    """
    ks_results = {}
    for col in original.columns:
        ks_stat, p_value = ks_2samp(original[col], synthetic[col])
        ks_results[col] = {'KS Statistic': ks_stat, 'p-value': p_value}
    return ks_results

def ks_test(original, synthetic):
    """Perform KS test to compare two distributions."""
    statistic, p_value = ks_2samp(original, synthetic)
    return statistic, p_value



if __name__ == "__main__":
    synthetic_datasets = ["copulagan", "ctgan", "gaussian_copula", "gmm", "tvae", "random"]
    original_datasets =["insurance", "insurance", "diabetes"]
    for orig in original_datasets:
        similarities = []
        for syn in synthetic_datasets:
            original_data = pd.read_csv(f"/Users/ksi/Development/Bachelorthesis/{orig}.csv")
            synthetic_data = pd.read_csv(
                f"/privacy_utility_framework/synprivutil/models/{orig}_datasets/{syn}_sample.csv")
            transformed_data_o, transformed_data_s = transform_and_normalize(original_data, synthetic_data)

            print(f"{syn}: {transformed_data_o.columns, transformed_data_s.columns}")
            ks_results = ks_test_columns(transformed_data_o, transformed_data_s)
            mean_ks_similarity = np.mean([1 - result['KS Statistic'] for result in ks_results.values()])
            similarities.append(mean_ks_similarity)

        print(f"SIMS {orig}")
        print(similarities)
        plt.figure(figsize=(10, 5))
        plt.bar(synthetic_datasets, similarities, color='skyblue')
        plt.xlabel('Synthetic Datasets')
        plt.ylabel('KS Similarity')
        plt.title(f'Mean KS Similarity for Different Synthetic Datasets (Original: {orig})')
        plt.ylim(0, 1)  #
        plt.show()
