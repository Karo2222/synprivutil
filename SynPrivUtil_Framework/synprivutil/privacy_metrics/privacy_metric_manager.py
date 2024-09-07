
from SynPrivUtil_Framework.synprivutil.privacy_metrics.privacy_metric_calculator import PrivacyMetricCalculator
from typing import Dict, Type


class PrivacyMetricManager:
    def __init__(self, original_data, synthetic_data):
        # Initialize shared parameters
        self.original_data = original_data
        self.synthetic_data = synthetic_data

        self.metric_instances = []

    def add_metric(self, metric_class: Type[PrivacyMetricCalculator], **kwargs):
        # Ensure the provided class is a subclass of PrivacyMetricEvaluator
        if not issubclass(metric_class, PrivacyMetricCalculator):
            raise TypeError("Metric class must be a subclass of PrivacyMetricCalculator.")

        try:
            metric_instance = metric_class(self.original_data, self.synthetic_data, **kwargs)
            self.metric_instances.append(metric_instance)
        except Exception as e:
            print(f"Error creating metric instance: {e}")

    def evaluate_all(self) -> Dict[str, float]:
        results = {}
        for metric in self.metric_instances:
            metric_name = metric.__class__.__name__
            results[metric_name] = metric.evaluate()
        return results
