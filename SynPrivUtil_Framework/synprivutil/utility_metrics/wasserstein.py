import pandas as pd
from scipy.stats import wasserstein_distance_nd

from SynPrivUtil_Framework.synprivutil.utility_metrics import UtilityMetricCalculator


class WassersteinCalculator(UtilityMetricCalculator):
    def __init__(self, original_data: pd.DataFrame, synthetic_data: pd.DataFrame):
        super().__init__(original_data, synthetic_data)

    def evaluate(self):
        distance = wasserstein_distance_nd(self.original_data, self.synthetic_data)
        return distance

