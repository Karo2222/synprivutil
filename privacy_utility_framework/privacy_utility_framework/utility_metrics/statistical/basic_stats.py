import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from privacy_utility_framework.privacy_utility_framework.models.transform import transform_and_normalize

#BasicStatsCalculator
def compute_basic_stats(orig, syn):
    stats = {}
    mean_diffs = []
    median_diffs = []
    var_diffs = []
    for col in orig.columns:
        orig_mean, syn_mean = orig[col].mean(), syn[col].mean()
        orig_median, syn_median = orig[col].median(), syn[col].median()
        orig_var, syn_var = orig[col].var(), syn[col].var()
        # print(f'Mean for {col}: orig={orig_mean}, syn={syn_mean}')
        # print(f'Median for {col}: orig={orig_median}, syn={syn_median}')
        # print(f'Variance for {col}: orig={orig_var}, syn={syn_var}')
        mean_diff = abs(syn_mean - orig_mean)
        median_diff = abs(syn_median - orig_median)
        var_diff = abs(syn_var - orig_var)

        mean_diffs.append(mean_diff)
        median_diffs.append(median_diff)
        var_diffs.append(var_diff)
        stats[col] = {
            'orig_mean': orig_mean, 'syn_mean': syn_mean,
            'orig_median': orig_median, 'syn_median': syn_median,
            'orig_var': orig_var, 'syn_var': syn_var
        }
    mean_score = np.mean(mean_diffs)
    median_score = np.mean(median_diffs)
    var_score = np.mean(var_diffs)
    print(f"Mean, Med, Var: {np.round(mean_score, 4), np.round(median_score, 4), np.round(var_score, 4)}")
    return stats

def compute_descriptive_stats(df: pd.DataFrame):
    stats = pd.DataFrame(index=df.columns)
    stats['mean'] = df.mean()
    stats['std'] = df.std()
    stats['min'] = df.min()
    stats['25%'] = df.quantile(0.25)
    stats['50%'] = df.median()
    stats['75%'] = df.quantile(0.75)
    stats['max'] = df.max()
    return stats

def correlate_descriptive_stats(orig, syn):
    orig_stats = compute_descriptive_stats(orig)
    syn_stats = compute_descriptive_stats(syn)
    correlations = {}
    for stat in orig_stats.columns:
        correlations[stat] = orig_stats[stat].corr(syn_stats[stat])
    return correlations

# original_data = pd.read_csv(f"/diabetes.csv")
# synthetic_data = pd.read_csv(f"/SynPrivUtil_Framework/synprivutil/models/diabetes_datasets/ctgan_sample.csv")
# trans_o, trans_s = transform_and_normalize(original_data, synthetic_data)
# print(trans_s)
# correlations = correlate_descriptive_stats(trans_o, trans_s)
# print(correlations)
# compute_basic_stats(trans_o, trans_s)


def plot_all_stats_for_stat(all_stats, stat_name, orig, syn):
    plt.figure(figsize=(12, 6))

    for label, stats in all_stats.items():
        syn_values = [values[f'syn_{stat_name}'] for values in stats.values()]
        columns = stats.keys()

        plt.plot(columns, syn_values, label=label, marker='o')

    # Use the first dataset's original values for reference
    first_label = list(all_stats.keys())[0]
    orig_values = [all_stats[first_label][col][f'orig_{stat_name}'] for col in columns]
    plt.plot(columns, orig_values, label='Original', marker='x', linestyle='--', linewidth=2)

    plt.xlabel('Columns')
    plt.ylabel(stat_name.capitalize())
    plt.title(f'Comparison of {stat_name.capitalize()} for Original ({orig}) and Synthetic Datasets')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{orig}_{stat_name.capitalize()}_basic_stats.png", bbox_inches='tight')
    plt.show()

def calculate_score(all_stats, stat_type):
    diffs = []
    for key in all_stats:
        for col, stats in all_stats[key].items():
            if stat_type == 'mean':
                diffs.append(abs(stats['syn_mean'] - stats['orig_mean']))
            elif stat_type == 'median':
                diffs.append(abs(stats['syn_median'] - stats['orig_median']))
            elif stat_type == 'var':
                diffs.append(abs(stats['syn_var'] - stats['orig_var']))
    return np.mean(diffs)
synthetic_datasets = ["copulagan", "ctgan", "gaussian_copula", "gmm", "tvae", "random"]
original_datasets =["diabetes", "cardio", "insurance"]

for orig in original_datasets:
    all_stats = {}
    for syn in synthetic_datasets:
        original_data = pd.read_csv(f"/Users/ksi/Development/Bachelorthesis/{orig}.csv")
        synthetic_data = pd.read_csv(f"/privacy_utility_framework/synprivutil/models/{orig}_datasets/{syn}_sample.csv")
        trans_o, trans_s = transform_and_normalize(original_data, synthetic_data)
        print(f"PAIR {orig, syn}")
        all_stats[f'{orig}_{syn}'] = compute_basic_stats(trans_o, trans_s)
    #print(all_stats)
    for stat_type in ['mean', 'median', 'var']:
        final_score = calculate_score(all_stats, stat_type)
        #print(f'Final {stat_type} score for {orig}: {final_score}')
    for stat in ['mean', 'median', 'var']:
        plot_all_stats_for_stat(all_stats, stat, orig, syn)


