import numpy as np
from scipy.spatial import distance


def calculate_min_distances(source, target, metric='euclidean'):
    """
    Calculate the minimum distances from each source sample to target samples
    and from each target sample to source samples using a specified distance metric.

    Parameters:
    source (numpy.ndarray): Source distribution samples (n_s x d).
    target (numpy.ndarray): Target distribution samples (n_t x d).
    metric (str): Distance metric to use (default is 'euclidean').

    Returns:
    tuple: (min_d_st, min_d_ts, min_d_ss, min_d_tt)
           - min_d_st: Minimum distance from each target sample to source samples.
           - min_d_ts: Minimum distance from each source sample to target samples.
           - min_d_ss: Minimum leave-one-out distance within source samples.
           - min_d_tt: Minimum leave-one-out distance within target samples.
    """
    # Calculate distances from target to source
    d_st = distance.cdist(target, source, metric=metric)
    min_d_st = np.min(d_st, axis=1)

    # Calculate distances from source to target
    d_ts = distance.cdist(source, target, metric=metric)
    min_d_ts = np.min(d_ts, axis=1)

    # Calculate distances within source samples (leave-one-out)
    d_ss = distance.cdist(source, source, metric=metric)
    np.fill_diagonal(d_ss, np.inf)  # Ignore self-distances
    min_d_ss = np.min(d_ss, axis=1)

    # Calculate distances within target samples (leave-one-out)
    d_tt = distance.cdist(target, target, metric=metric)
    np.fill_diagonal(d_tt, np.inf)  # Ignore self-distances
    min_d_tt = np.min(d_tt, axis=1)

    return min_d_st, min_d_ts, min_d_ss, min_d_tt


def nnaa(source, target, metric='euclidean'):
    """
    Calculate the Nearest Neighbor Adversarial Accuracy (NNAA).

    Parameters:
    source (numpy.ndarray): Source distribution samples (n_s x d).
    target (numpy.ndarray): Target distribution samples (n_t x d).
    metric (str): Distance metric to use (default is 'euclidean').

    Returns:
    float: The calculated NNAA.
    """

    # TODO: make sure it is normalized

    # Calculate minimum distances
    min_d_st, min_d_ts, min_d_ss, min_d_tt = calculate_min_distances(source, target, metric)

    # NOTE: Based on paper, n_s = n_t
    # Apply the formula
    n_s = len(source)
    n_t = len(target)

    term1 = np.sum(min_d_ts >= min_d_tt) / n_t
    term2 = np.sum(min_d_st >= min_d_ss) / n_s

    nnaa_value = 0.5 * (term1 + term2)

    return nnaa_value

# Example usage
source = np.random.rand(100, 5)  # 100 samples, 5 features each (Source distribution)
target = np.random.rand(100, 5)  # 100 samples, 5 features each (Target distribution)

nnaa_value = nnaa(source, target)
#print(f"Nearest Neighbor Adversarial Accuracy (NNAA): {nnaa_value}")