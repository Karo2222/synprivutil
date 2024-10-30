from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error

from privacy_utility_framework.privacy_utility_framework.models.transform import transform_rdt, transform_and_normalize
from privacy_utility_framework.privacy_utility_framework.utility_metrics import MICalculator, KSCalculator, \
    CorrelationCalculator, CorrelationMethod


def plot_original_vs_synthetic(original_data, synthetic_data):
    num_columns = len(original_data.columns)
    fig, axes = plt.subplots(nrows=num_columns, ncols=1, figsize=(10, 5 * num_columns))

    for i, column in enumerate(original_data.columns):
        sns.kdeplot(original_data[column], ax=axes[i], label='Original', color='blue')
        sns.kdeplot(synthetic_data[column], ax=axes[i], label='Synthetic', color='red')
        axes[i].set_title(f'Distribution of {column}')
        axes[i].legend()

    plt.tight_layout()
    plt.show()

def plot_pairwise_relationships(original_data, synthetic_data, title):
    # Combine the datasets for comparison
    original_data['Dataset'] = 'Original'
    synthetic_data['Dataset'] = 'Synthetic'
    combined_data = pd.concat([original_data, synthetic_data])

    # Create pairplot
    sns.pairplot(combined_data, hue='Dataset', plot_kws={'alpha': 0.5})
    plt.suptitle(title, y=1.02)
    plt.show()

def mutual_information_heatmap(original_data, synthetic_data, figure_filepath, orig, syn, attributes: List = None):
    if attributes:
        orig_df = original_data[attributes]
        syn_df = synthetic_data[attributes]
    else:
        orig_df = original_data
        syn_df = synthetic_data

    private_mi = MICalculator.pairwise_attributes_mutual_information(orig_df)
    synthetic_mi = MICalculator.pairwise_attributes_mutual_information(syn_df)

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


def correlation_plot_heatmap(original_data, synthetic_data, original_name, synthetic_name):
        calc = CorrelationCalculator(original_data, synthetic_data, original_name=original_name, synthetic_name=synthetic_name)
        method = CorrelationMethod.PEARSON
        orig_corr, syn_corr = calc.correlation_pairs(method)
        fig = plt.figure(figsize=(15, 6), dpi=120)
        fig.suptitle(f'{method} Correlation Comparison (Original vs Synthetic)', fontsize=20)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        sns.heatmap(orig_corr, ax=ax1, cmap="GnBu")
        sns.heatmap(syn_corr, ax=ax2, cmap="GnBu")
        ax1.set_title(f'Original Data Correlation, {calc.original.name} dataset', fontsize=15)
        ax2.set_title(f'Synthetic Data Correlation, generated with {calc.synthetic.name}', fontsize=15)
        fig.autofmt_xdate()
        fig.tight_layout()
        plt.subplots_adjust(top=0.83)
        plt.show()
