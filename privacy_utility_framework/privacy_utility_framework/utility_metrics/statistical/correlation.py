import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr
from dython.nominal import theils_u
from sklearn.metrics import mean_squared_error
import seaborn as sns

from privacy_utility_framework.privacy_utility_framework.models.transform import transform_and_normalize

# NOTE: really small values
#CorrelationCalculator
def correlation_coefficients(original, synthetic):
    """
    Calculate Pearson and Spearman correlation coefficients between corresponding features.

    Parameters:
        original (pd.DataFrame): Original data.
        synthetic (pd.DataFrame): Synthetic data.

    Returns:
        dict: Pearson and Spearman correlation coefficients for each feature.
    """

    # TODO: change the code to look at the correlation between two orig columns,
    #  and compare that correlation to corr. between two syn columns
    corr_results = {}
    for col in original.columns:
        pearson_corr, _ = pearsonr(original[col], synthetic[col])
        spearman_corr, _ = spearmanr(original[col], synthetic[col])
        corr_results[col] = {'Pearson': pearson_corr, 'Spearman': spearman_corr}
    return corr_results

def compute_column_similarity(orig, syn):
    column_similarities = {}
    for col in orig.columns:
        if orig[col].dtype in ['int64', 'float64']:
            corr, _ = pearsonr(orig[col], syn[col])
            column_similarities[col] = corr
        else:
            column_similarities[col] = theils_u(orig[col], syn[col])
    return column_similarities

if __name__ == "__main__":
    synthetic_datasets = ["copulagan", "ctgan", "gaussian_copula", "gmm", "tvae", "random"]
    original_datasets =["diabetes", "cardio", "insurance"]

    for orig in original_datasets:
        for syn in synthetic_datasets:
            original_data = pd.read_csv(f"/Users/ksi/Development/Bachelorthesis/{orig}.csv")
            synthetic_data = pd.read_csv(
                f"/privacy_utility_framework/synprivutil/models/{orig}_datasets/{syn}_sample.csv")
            trans_o, trans_s = transform_and_normalize(original_data, synthetic_data)
            method = 'spearman'
            orig_corr = trans_o.corr(method=method)
            syn_corr = trans_s.corr(method=method)
            #print(orig_corr, syn_corr)
            # Plot the heatmaps
            fig = plt.figure(figsize=(15, 6), dpi=120)
            fig.suptitle(f'{method} Correlation Comparison (Original vs Synthetic)', fontsize=20)
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            sns.heatmap(orig_corr, ax=ax1, cmap="GnBu")
            sns.heatmap(syn_corr, ax=ax2, cmap="GnBu")
            ax1.set_title(f'Original Data Correlation, {orig} dataset', fontsize=15)
            ax2.set_title(f'Synthetic Data Correlation, generated with {syn}', fontsize=15)
            fig.autofmt_xdate()
            fig.tight_layout()
            plt.subplots_adjust(top=0.83)
            plt.savefig(f"{orig}_{syn}_{method}_correlation_heatmaps.png", bbox_inches='tight')
            #plt.show()
            # Flatten the matrices
            orig_corr_flat = orig_corr.to_numpy().flatten()
            syn_corr_flat = syn_corr.to_numpy().flatten()
            score = 1 - abs(syn_corr_flat - orig_corr_flat) / 2
            #print(f"score {score}")
            rmse = np.sqrt(mean_squared_error(orig_corr_flat, syn_corr_flat))
            print(round(np.mean(score), 4))
        #print(f'{orig, syn}: RMSE: {round(rmse, 4)}')
# original_data = pd.read_csv(f"/Users/ksi/Development/Bachelorthesis/insurance.csv")
# synthetic_data = pd.read_csv(f"/Users/ksi/Development/Bachelorthesis/SynPrivUtil_Framework/synprivutil/models/insurance_datasets/random_sample.csv")
# transformed_data_o, transformed_data_s = transform_and_normalize(original_data, synthetic_data)
#
# corr_results = correlation_coefficients(transformed_data_o, transformed_data_s)
# print(f"CORRRR RES {corr_results}")
# synthetic_datasets = ["copulagan", "ctgan", "gaussian_copula", "gmm", "tvae", "random"]
# original_datasets =["cardio", "insurance", "diabetes"]
# for orig in original_datasets:
#     all_correlations = {method: [] for method in ['Pearson', 'Spearman']}
#     for syn in synthetic_datasets:
#         original_data = pd.read_csv(f"/Users/ksi/Development/Bachelorthesis/{orig}.csv")
#         synthetic_data = pd.read_csv(f"/Users/ksi/Development/Bachelorthesis/SynPrivUtil_Framework/synprivutil/models/{orig}_datasets/{syn}_sample.csv")
#         transformed_data_o, transformed_data_s = transform_and_normalize(original_data, synthetic_data)
#
#         corr_results = correlation_coefficients(transformed_data_o, transformed_data_s)
#         mean_pearson = np.mean([corr['Pearson'] for corr in corr_results.values()])
#         mean_spearman = np.mean([corr['Spearman'] for corr in corr_results.values()])
#         all_correlations['Pearson'].append(mean_pearson)
#         all_correlations['Spearman'].append(mean_spearman)
#     print(f"{all_correlations}")
#
#
#     bar_width = 0.35  # the width of the bars
#     index = np.arange(len(synthetic_datasets))
#
#     fig, ax = plt.subplots(figsize=(12, 6))
#
#     plt.bar(index, all_correlations['Pearson'], bar_width, label='Pearson', color='skyblue')
#     plt.bar(index + bar_width, all_correlations['Spearman'], bar_width, label='Spearman', color='salmon')
#
#     plt.xlabel('Synthetic Datasets')
#     plt.ylabel('Mean Correlation')
#     plt.title('Mean Pearson & Spearman Correlations for Different Synthetic Datasets (Original: YourOriginalDatasetName)')
#     plt.xticks(index + bar_width / 2, synthetic_datasets)
#     plt.legend()
#     plt.ylim(-1, 1)  # Because correlation ranges from -1 to 1
#
#     plt.show()
