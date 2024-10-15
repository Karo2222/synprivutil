import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import wasserstein_distance_nd
import ot

from SynPrivUtil_Framework.synprivutil.models.transform import transform_and_normalize
from SynPrivUtil_Framework.synprivutil.utility_metrics import UtilityMetricCalculator

from enum import Enum

class WassersteinMethod(Enum):
    #SINKHORN = "sinkhorn"
    #WASSERSTEIN = "wasserstein"
    WASSERSTEIN_SAMPLE = "wasserstein_sample"


class WassersteinCalculator(UtilityMetricCalculator):
    def __init__(self, original_data: pd.DataFrame, synthetic_data: pd.DataFrame):
        super().__init__(original_data, synthetic_data)

    def evaluate(self, metric=WassersteinMethod.WASSERSTEIN_SAMPLE, n_samples=500, n_iterations=1):
        # if metric == WassersteinMethod.SINKHORN:
        #     numItermax = 1000
        #     stopThr = 1e-9
        #     orig_np = self.original_data.to_numpy()
        #     syn_np = self.synthetic_data.to_numpy()
        #
        #     # Compute pairwise distance matrix
        #     M = ot.dist(orig_np, syn_np, metric="euclidean")
        #     reg = 0.0025
        #     # Compute Sinkhorn approximation of Wasserstein distance
        #     wass_dist = ot.sinkhorn2(np.ones((orig_np.shape[0],)) / orig_np.shape[0],
        #                              np.ones((syn_np.shape[0],)) / syn_np.shape[0],
        #                              M, reg, stopThr=stopThr, numItermax=numItermax)
        #     print(f'Sinkhorn Wasserstein Distance: {wass_dist}')
        #     return wass_dist
        # elif metric == WassersteinMethod.WASSERSTEIN:
        #     distance = wasserstein_distance_nd(self.original_data, self.synthetic_data)
        #     print(f"Wasserstein was: {distance}")
        #     return distance
        if metric == WassersteinMethod.WASSERSTEIN_SAMPLE:
            distances = []
            # Loop over the number of iterations
            for _ in range(n_iterations):
                # Randomly sample a subset of the data
                orig_sample = self.original_data.sample(n=n_samples, random_state=np.random.randint(0, 10000))
                syn_sample = self.synthetic_data.sample(n=n_samples, random_state=np.random.randint(0, 10000))

                # Compute the Wasserstein distance for the sample
                dist = wasserstein_distance_nd(orig_sample, syn_sample)
                distances.append(dist)
            sampled_dist = np.mean(distances)
            print(
                f"Sampled Wasserstein distance was: {sampled_dist}, with n_samples={n_samples}, n_iterations={n_iterations}")
            # Return the average of all computed distances
            return sampled_dist

# original_data = pd.read_csv("/Users/ksi/Development/Bachelorthesis/diabetes.csv")
# synthetic_data = pd.read_csv("/Users/ksi/Development/Bachelorthesis/SynPrivUtil_Framework/synprivutil/models/diabetes_datasets/ctgan_sample.csv")
#
# transformed_data_o, transformed_data_s = transform_and_normalize(original_data, synthetic_data)
# test_wasserstein_calc = WassersteinCalculator(transformed_data_o, transformed_data_s)
# t = test_wasserstein_calc.evaluate(WassersteinMethod.WASSERSTEIN_SAMPLE, 50, 10)
# print(f"wasserstein sample {t}")
# t = test_wasserstein_calc.evaluate(WassersteinMethod.WASSERSTEIN)
# print(f"wasserstein {t}")
# t = test_wasserstein_calc.evaluate(WassersteinMethod.SINKHORN)
# print(f"sinkhorn {t}")

if __name__ == "__main__":
    synthetic_datasets = ["copulagan", "ctgan", "gaussian_copula", "gmm", "tvae", "random"]
    original_datasets =["cardio"]
    all_wasserstein_distances = {method: [] for method in WassersteinMethod}
    print(all_wasserstein_distances)
    for orig in original_datasets:
        for syn in synthetic_datasets:
            print(f"PAIR {orig} {syn}")
            original_data = pd.read_csv(f"/Users/ksi/Development/Bachelorthesis/{orig}.csv")
            synthetic_data = pd.read_csv(f"/Users/ksi/Development/Bachelorthesis/SynPrivUtil_Framework/synprivutil/models/{orig}_datasets/{syn}_sample.csv")
            transformed_data_o, transformed_data_s = transform_and_normalize(original_data, synthetic_data)
            for method in all_wasserstein_distances:
                calc = WassersteinCalculator(transformed_data_o, transformed_data_s)
                res = calc.evaluate(metric=method, n_samples=1300)
                all_wasserstein_distances[method].append(res)

        # # Plot the distances
        # bar_width = 0.35  # the width of the bars
        # index = np.arange(len(synthetic_datasets))
        #
        # fig, ax = plt.subplots(figsize=(12, 6))
        #
        # for i, method in enumerate(WassersteinMethod):
        #     plt.bar(index + i * bar_width, all_wasserstein_distances[method], bar_width, label=method.value)
        #
        # plt.xlabel('Synthetic Datasets')
        # plt.ylabel('Distance')
        # plt.title(f'Wasserstein, Wasserstein Sample and Sinkhorn Distances for Different Synthetic Datasets (Original: {orig})')
        # plt.xticks(index + bar_width / 2, synthetic_datasets)
        # plt.legend()
        # plt.ylim(0, max([max(vals) for vals in all_wasserstein_distances.values()]) * 1.1)  # Adjust y-axis limit for better visualization
        #
        # plt.show()