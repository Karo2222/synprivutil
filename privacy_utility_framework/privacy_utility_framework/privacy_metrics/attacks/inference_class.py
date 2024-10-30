from typing import Optional
import pandas as pd
from anonymeter.evaluators import InferenceEvaluator
from privacy_utility_framework.privacy_utility_framework.privacy_metrics import PrivacyMetricCalculator


# DONE
class InferenceCalculator(PrivacyMetricCalculator):
    def __init__(self, original: pd.DataFrame, synthetic: pd.DataFrame,
                 aux_cols: list[str], secret: str,
                 regression: Optional[bool] = None,
                 n_attacks: int = 500,
                 control: Optional[pd.DataFrame] = None,
                 original_name: str = None, synthetic_name: str = None,):

        """
        Initializes the InferenceCalculator instance for evaluating inference risk.

        Parameters:
        - original (pd.DataFrame): The original dataset.
        - synthetic (pd.DataFrame): The synthetic dataset generated from the original data.
        - aux_cols (list[str]): Features of the records that are given to the attacker as auxiliary information.
        - secret (str): The name of the secret column to be inferred.
        - regression (Optional[bool]): Specifies whether the target of the inference attack is quantitative
            (regression = True) or categorical (regression = False). If None (default),
            the code will try to guess this by checking the type of the variable
        - n_attacks (int): The number of inference attacks to perform. Defaults to 500, limited by the control dataset size.
        - control (Optional[pd.DataFrame]): An optional control dataset for evaluating inference risk.
        - original_name (Optional[str]): An optional name for the original dataset.
        - synthetic_name (Optional[str]): An optional name for the synthetic dataset.

        Raises:
        ValueError: If aux_cols or secret parameters are not provided.
        """
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
        # Retrieve the data from the original and synthetic Dataset objects (no need for normalization or
        # transformation)
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
        # Perform the evaluation and return the calculated risk
        return evaluator.evaluate().risk()







