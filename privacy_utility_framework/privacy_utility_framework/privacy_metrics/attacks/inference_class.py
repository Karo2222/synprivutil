from typing import Optional
import pandas as pd
from anonymeter.evaluators import InferenceEvaluator
from privacy_utility_framework.privacy_utility_framework.privacy_metrics import PrivacyMetricCalculator


class InferenceCalculator(PrivacyMetricCalculator):
    def __init__(self, original: pd.DataFrame, synthetic: pd.DataFrame,
                 aux_cols: list[str], secret: str,
                 regression: Optional[bool] = None,
                 n_attacks: int = 500,
                 control: Optional[pd.DataFrame] = None,
                 original_name: str = None, synthetic_name: str = None,):
        super().__init__(original, synthetic, aux_cols=aux_cols,
                         original_name=original_name, synthetic_name=synthetic_name)
        if aux_cols is None:
            raise ValueError("Parameter 'aux_cols' is required in InferenceCalculator.")
        if secret is None:
            raise ValueError("Parameter 'secret' is required in InferenceCalculator.")
        self.aux_cols = aux_cols
        self.secret = secret
        self.regression = regression
        self.n_attacks = min(n_attacks, len(control))
        self.control = control

    def evaluate(self):
        """
        For a thorough interpretation of the attack result,
        it is recommended to set aside a small portion of the
        original dataset to use as a control dataset for the
        Inference Attack. These control records should not have
        been used to generate the synthetic dataset.
        For good statistical accuracy on the attack results,
        500 to 1000 control records are usually enough.
        """
        # The original data is used for the Inference Risk (no need to normalize or transform).
        original = self.original.data
        synthetic = self.synthetic.data
        evaluator = InferenceEvaluator(
            ori=original,
            syn=synthetic,
            aux_cols=self.aux_cols,
            secret=self.secret,
            regression=self.regression,
            n_attacks=self.n_attacks,
            control=self.control
        )
        # Perform the evaluation and return the result
        return evaluator.evaluate().risk()







