import pandas as pd


class UtilityMetricCalculator:
    def __init__(self, original_data: pd.DataFrame, synthetic_data: pd.DataFrame, **kwargs):
        self.original_data = original_data
        self.synthetic_data = synthetic_data

    def _handle_additional_inputs(self, **kwargs):
        pass

    def evaluate(self) -> float:
        raise NotImplementedError("Subclasses should implement this method.")

