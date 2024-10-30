import numpy as np
import ot
import pandas as pd
from scipy.stats import wasserstein_distance_nd

from privacy_utility_framework.privacy_utility_framework.utility_metrics import ks_test_columns, correlation_coefficients
from privacy_utility_framework.privacy_utility_framework.utility_metrics.statistical import mutual_information

if __name__ == "__main__":
    original_data = pd.read_csv('/Users/ksi/Development/Bachelorthesis/SD2011_selected_columns.csv')
    synthetic_data = pd.read_csv('/Users/ksi/Development/Bachelorthesis/syn_SD2011_selected_columns.csv')

    original_data  = pd.read_csv("/Users/ksi/Development/Bachelorthesis/diabetes_transformed_ctgan.csv")
    synthetic_data = pd.read_csv("/Users/ksi/Development/Bachelorthesis/synthetic_data_transformed_ctgan.csv")


    # NOTE: Error, contains NaN
    ks_results = ks_test_columns(original_data, synthetic_data)
    mi_results = mutual_information(original_data, synthetic_data)
    corr_results = correlation_coefficients(original_data, synthetic_data)

    # Display Results
    print("Kolmogorov-Smirnov Test Results:")
    print(ks_results)

    print("\nMutual Information Results:")
    print(mi_results)

    print("\nCorrelation Coefficient Results:")
    print(corr_results)


    # Example data
    original_data = np.random.rand(1000, 3)  # 100 samples, 3 dimensions
    synthetic_data = np.random.rand(1000, 3)

    # Compute the Wasserstein distance for the entire multi-dimensional data
    #distance = wasserstein_distance_nd(original_data, synthetic_data)

    # Compute the cost matrix (Euclidean distance between points)
    cost_matrix = ot.dist(original_data, synthetic_data)

    # Compute the Sinkhorn distance with a higher regularization term for faster computation
    sinkhorn_distance = ot.sinkhorn2(np.ones((1000,)) / 1000, np.ones((1000,)) / 1000, cost_matrix, reg=1e-3)
    print("Sinkhorn distance:", sinkhorn_distance)

    #print("Wasserstein distance (N-dimensional):", distance)

    # # Example data
    # original_data = np.random.rand(5000, 3)
    # synthetic_data = np.random.rand(5000, 3)

    # # Reduce dimensionality
    # pca = PCA(n_components=2)
    # original_reduced = pca.fit_transform(original_data)
    # synthetic_reduced = pca.transform(synthetic_data)
    #
    # # Compute the Wasserstein distance for the reduced data
    # distance = wasserstein_distance_nd(original_reduced, synthetic_reduced)
    #
    # print("Wasserstein distance (reduced dimensions):", distance)

    # Example data
    original_data = np.random.rand(1000, 3)
    synthetic_data = np.random.rand(1000, 3)

    # Compute the Sinkhorn distance
    cost_matrix = ot.dist(original_data, synthetic_data)
    sinkhorn_dist = ot.sinkhorn2(np.ones((1000,)) / 1000, np.ones((1000,)) / 1000, cost_matrix, reg=1e-3, numItermax=10000)
    print("Sinkhorn distance:", sinkhorn_dist)
    # Compute the Wasserstein distance (N-dimensional)
    wasserstein_dist = wasserstein_distance_nd(original_data, synthetic_data)


    print("Wasserstein distance (N-dimensional):", wasserstein_dist)
