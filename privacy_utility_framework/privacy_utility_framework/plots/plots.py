import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from privacy_utility_framework.privacy_utility_framework.models.transform import transform_rdt


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


original_data = pd.read_csv('/datasets/original/diabetes.csv')
synthetic_data = pd.read_csv('/datasets/synthetic/diabetes_datasets/ctgan_sample.csv')
transformed_orig, transformed_syn, _ = transform_rdt(original_data, synthetic_data)
plot_original_vs_synthetic(transformed_orig, transformed_syn)

plot_pairwise_relationships(transformed_orig, transformed_syn, 'Pairwise Relationships: Original vs Synthetic Data')