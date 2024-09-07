import pandas as pd
from anonymeter.evaluators import InferenceEvaluator

from SynPrivUtil_Framework.synprivutil.privacy_metrics import PrivacyMetricCalculator


class SinglingOutCalculator(PrivacyMetricCalculator):
    def __init__(self, original_data: pd.DataFrame, synthetic_data: pd.DataFrame, aux_cols: list[str], secret: str):
        super().__init__(original_data, synthetic_data)
        if aux_cols is None:
            raise ValueError("Parameter 'aux_cols' is required in SinglingOutCalculator.")
        if secret is None:
            raise ValueError("Parameter 'secret' is required in SinglingOutCalculator.")
        self.aux_cols = aux_cols
        self.secret = secret

    def evaluate(self):
        evaluator = InferenceEvaluator(
            ori=self.original_data,
            syn=self.synthetic_data,
            aux_cols=self.aux_cols,
            secret=self.secret
        )
        return evaluator.evaluate().risk()
