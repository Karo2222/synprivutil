import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp
from scipy.special import kl_div

from SynPrivUtil_Framework.synprivutil.models.transform import transform_rdt, transform_and_normalize

#JSCalculator
def kl_divergence2(original, synthetic):
    """Calculate KL Divergence between two datasets."""
    original_values = np.concatenate([original[col].values for col in original.columns])
    synthetic_values = np.concatenate([synthetic[col].values for col in synthetic.columns])

    hist_range = (min(original_values.min(), synthetic_values.min()), max(original_values.max(), synthetic_values.max()))
    original_hist, bin_edges = np.histogram(original_values, bins='auto', range=hist_range, density=True)
    synthetic_hist, _ = np.histogram(synthetic_values, bins=bin_edges, density=True)


    # Calculate KL Divergence
    kl_div_result = np.sum(kl_div(synthetic_hist, original_hist))
    return kl_div_result

def kl_divergence(original, synthetic):
    """Calculate KL Divergence between two datasets."""
    # Ensure the input is normalized to form a probability distribution
    original_hist, original_bins = np.histogram(original, bins='auto', density=True)
    synthetic_hist, synthetic_bins = np.histogram(synthetic, bins='auto', density=True)

    # To avoid division by zero, replace zeros with a small number
    original_hist = np.where(original_hist == 0, 1e-10, original_hist)
    synthetic_hist = np.where(synthetic_hist == 0, 1e-10, synthetic_hist)

    # Calculate KL Divergence
    kl_div_result = np.sum(kl_div(synthetic_hist, original_hist))
    return kl_div_result

# def compute_jsd(orig, syn):
#     for col in orig.columns:
#         jsd = jensenshannon(orig[col], syn[col])
#         print(f'Jensen-Shannon Divergence for {col}: {jsd}')

def compute_js_similarity(orig, syn, bins='auto'):
    js_similarities = {}
    for col in orig.columns:
        hist_range = (min(orig[col].min(), syn[col].min()), max(orig[col].max(), syn[col].max()))

        orig_hist, bin_edges = np.histogram(orig[col], bins=bins, range=hist_range, density=True)
        syn_hist, _ = np.histogram(syn[col], bins=bin_edges, density=True)

        # To avoid division by zero, replace zeros with a small number
        orig_hist = np.where(orig_hist == 0, 1e-10, orig_hist)
        syn_hist = np.where(syn_hist == 0, 1e-10, syn_hist)

        js_distance = jensenshannon(orig_hist, syn_hist)
        js_similarity = 1 - js_distance  # Higher score is better
        js_similarities[col] = js_similarity
    overall_js_similarity_score = np.mean(list(js_similarities.values()))
    print(f"OVERALL SCORE: {overall_js_similarity_score}")
    return js_similarities


original_data = pd.read_csv(f"/Users/ksi/Development/Bachelorthesis/insurance.csv")
synthetic_data = pd.read_csv(f"/Users/ksi/Development/Bachelorthesis/SynPrivUtil_Framework/synprivutil/models/insurance_datasets/ctgan_sample.csv")
trans_o, trans_s, _ = transform_rdt(original_data, synthetic_data)
print(trans_s.columns, trans_o.columns)
# Calculate metrics
kl_result = kl_divergence(trans_o, trans_s)
kl2 = kl_divergence2(trans_o, trans_s)

# Output results
print(f"KL Divergence 1: {kl_result}")
print(f"KL Divergence 2: {kl2}")

js_similarities = compute_js_similarity(trans_o, trans_s)
print(f"JS sim {js_similarities}")

#compute_jsd(trans_o, trans_s)

def plot_metrics(metrics):
    plt.figure(figsize=(12, 6))

    kl_divs = metrics['kl_divergence']
    js_sims = metrics['js_similarity']

    # Extract keys and values for plotting
    keys = list(kl_divs.keys())
    kl_values = [kl_divs[key] for key in keys]
    js_values = [np.mean(list(js_sims[key].values())) for key in keys]

    plt.subplot(1, 2, 1)
    plt.bar(keys, kl_values, color='skyblue')
    plt.xticks(rotation=45)
    plt.xlabel('Datasets')
    plt.ylabel('KL Divergence')
    plt.title('KL Divergence for Different Datasets')

    plt.subplot(1, 2, 2)
    plt.bar(keys, js_values, color='salmon')
    plt.xticks(rotation=45)
    plt.xlabel('Datasets')
    plt.ylabel('Mean JS Similarity')
    plt.title('JS Similarity for Different Datasets')

    plt.tight_layout()
    plt.show()
synthetic_datasets = ["copulagan", "ctgan", "gaussian_copula", "gmm", "tvae", "random"]
original_datasets = ["cardio", "insurance", "diabetes"]

metrics = {'kl_divergence': {}, 'js_similarity': {}}
for orig in original_datasets:
    metrics = {'kl_divergence': {}, 'js_similarity': {}}
    original_data = pd.read_csv(f"/Users/ksi/Development/Bachelorthesis/{orig}.csv")
    for syn in synthetic_datasets:
        synthetic_data = pd.read_csv(f"/Users/ksi/Development/Bachelorthesis/SynPrivUtil_Framework/synprivutil/models/{orig}_datasets/{syn}_sample.csv")
        trans_o, trans_s = transform_and_normalize(original_data, synthetic_data)
        key = f'{orig}_{syn}'
        print(f"PAIR {orig, syn}")
        #metrics['kl_divergence'][key] = kl_divergence2(trans_o, trans_s)
        metrics['js_similarity'][key] = compute_js_similarity(trans_o, trans_s)
    #plot_metrics(metrics)

