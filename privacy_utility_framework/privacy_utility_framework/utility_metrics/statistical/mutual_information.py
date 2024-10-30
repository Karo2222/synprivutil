from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score, mean_squared_error
import seaborn as sns


from privacy_utility_framework.privacy_utility_framework.models.transform import transform_and_normalize

#MICalculator
def get_column_types(df):
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    continuous_cols = df.select_dtypes(include=['int64', 'float64']).columns
    return categorical_cols, continuous_cols

def compute_hybrid_mutual_information(orig, syn):
    categorical_cols, continuous_cols = get_column_types(orig)
    mi_results = {}

    for col in categorical_cols:
        mi_score = mutual_info_score(orig[col], syn[col])
        mi_results[col] = mi_score

    for col in continuous_cols:
        mi_regression = mutual_info_regression(orig[[col]], syn[col])
        mi_results[col] = mi_regression[0]

    return mi_results

def pairwise_attributes_mutual_information(dataset):
    """Compute normalized mutual information for all pairwise attributes. Return a DataFrame."""
    sorted_columns = sorted(dataset.columns)
    mi_df = DataFrame(columns=sorted_columns, index=sorted_columns, dtype=float)
    for row in mi_df.columns:
        for col in mi_df.columns:
            mi_df.loc[row, col] = normalized_mutual_info_score(dataset[row].astype(str),
                                                               dataset[col].astype(str),
                                                               average_method='arithmetic')
    return mi_df

def mutual_information_heatmap(original_data, synthetic_data, figure_filepath, orig, syn, attributes: List = None):
    if attributes:
        orig_df = original_data[attributes]
        syn_df = synthetic_data[attributes]
    else:
        orig_df = original_data
        syn_df = synthetic_data

    private_mi = pairwise_attributes_mutual_information(orig_df)
    synthetic_mi = pairwise_attributes_mutual_information(syn_df)
    #print(synthetic_mi)
    private_mi_flat = private_mi.to_numpy().flatten()
    synthetic_mi_flat = synthetic_mi.to_numpy().flatten()

    # Calculate RMSE
    score = 1 - abs(synthetic_mi_flat - private_mi_flat) / 2
    rmse = np.sqrt(mean_squared_error(private_mi_flat, synthetic_mi_flat))
    print(round(np.mean(score), 4))
    #print(f'Pairwise mutual information, RMSE: {rmse}')

    fig = plt.figure(figsize=(15, 6), dpi=120)
    fig.suptitle('Pairwise Mutual Information Comparison (Original vs Synthetic)', fontsize=20)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    sns.heatmap(private_mi, ax=ax1, cmap="GnBu")
    sns.heatmap(synthetic_mi, ax=ax2, cmap="GnBu")
    ax1.set_title(f'Original Data NMI, {orig} dataset', fontsize=15)
    ax2.set_title(f'Synthetic Data NMI, generated with {syn}', fontsize=15)
    fig.autofmt_xdate()
    fig.tight_layout()
    plt.subplots_adjust(top=0.83)
    plt.savefig(figure_filepath, bbox_inches='tight')
    plt.show()

    #plt.close()

# original_data = pd.read_csv("/Users/ksi/Development/Bachelorthesis/diabetes.csv")
# synthetic_data = pd.read_csv("/Users/ksi/Development/Bachelorthesis/SynPrivUtil_Framework/synprivutil/models/diabetes_datasets/ctgan_sample.csv")
#
# r = compute_hybrid_mutual_information(original_data, synthetic_data)
# print(r)
# mutual_information_heatmap(original_data, synthetic_data, "test")

if __name__ == "__main__":
    synthetic_datasets = ["copulagan", "ctgan", "gaussian_copula", "gmm", "tvae", "random"]
    original_datasets =["diabetes", "cardio", "insurance"]

    for orig in original_datasets:
        for syn in synthetic_datasets:
            original_data = pd.read_csv(f"/Users/ksi/Development/Bachelorthesis/{orig}.csv")
            synthetic_data = pd.read_csv(
                f"/privacy_utility_framework/synprivutil/models/{orig}_datasets/{syn}_sample.csv")

            mutual_information_heatmap(original_data, synthetic_data, f"{orig}_{syn}_pairwise_norm_mi.png", orig, syn)