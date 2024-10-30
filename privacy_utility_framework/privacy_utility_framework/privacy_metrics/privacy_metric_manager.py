import pandas as pd

from privacy_utility_framework.privacy_utility_framework.privacy_metrics import DCRCalculator, NNDRCalculator
from privacy_utility_framework.privacy_utility_framework.privacy_metrics.distance.adversarial_accuracy_class import \
    AdversarialAccuracyCalculator
from privacy_utility_framework.privacy_utility_framework.privacy_metrics.privacy_metric_calculator import PrivacyMetricCalculator
from typing import Union, List, Dict


class PrivacyMetricManager:
    def __init__(self):
        self.metric_instances = []

    def add_metric(self, metric_instance: Union[PrivacyMetricCalculator, List[PrivacyMetricCalculator]]):
        # Check if the input is a single instance or a list of instances
        if isinstance(metric_instance, list):
            for metric in metric_instance:
                self._add_single_metric(metric)
        else:
            self._add_single_metric(metric_instance)

    def _add_single_metric(self, metric_instance: PrivacyMetricCalculator):
        # Ensure the provided class is a subclass of PrivacyMetricCalculator
        if not isinstance(metric_instance, PrivacyMetricCalculator):
            raise TypeError("Metric class must be a subclass of PrivacyMetricCalculator.")

        try:
            self.metric_instances.append(metric_instance)
        except Exception as e:
            print(f"Error creating metric instance: {e}")

    def evaluate_all(self) -> Dict[str, float]:
        results = {}
        for metric in self.metric_instances:
            metric_name = metric.__class__.__name__
            results[metric_name] = metric.evaluate()
        return results



if __name__ == "__main__":
    # Evaluate all added metrics
    original_data = pd.read_csv(f"/Users/ksi/Development/Bachelorthesis/datasets/original/diabetes.csv")
    synthetic_data = pd.read_csv(
    f"/Users/ksi/Development/Bachelorthesis/datasets/synthetic/diabetes_datasets/ctgan_sample.csv")
    p = PrivacyMetricManager()
    metric_list = [DCRCalculator(original_data, synthetic_data), NNDRCalculator(original_data, synthetic_data), AdversarialAccuracyCalculator(original_data, synthetic_data)]
    p.add_metric(metric_list)
    results = p.evaluate_all()
    for key, value in results.items():
        print(f"{key}: {value}")