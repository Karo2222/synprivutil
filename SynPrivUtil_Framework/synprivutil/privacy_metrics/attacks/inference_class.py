from typing import List, Optional

import pandas as pd
from anonymeter.evaluators import InferenceEvaluator

from SynPrivUtil_Framework.synprivutil.models.transform import transform_and_normalize
from SynPrivUtil_Framework.synprivutil.privacy_metrics import PrivacyMetricCalculator


class InferenceCalculator(PrivacyMetricCalculator):
    def __init__(self, original_data: pd.DataFrame, synthetic_data: pd.DataFrame,
                 aux_cols: list[str], secret: str,
                 regression: Optional[bool] = None,
                 n_attacks: int = 500,
                 control: Optional[pd.DataFrame] = None):
        super().__init__(original_data, synthetic_data, aux_cols=aux_cols)
        if aux_cols is None:
            raise ValueError("Parameter 'aux_cols' is required in InferenceCalculator.")
        if secret is None:
            raise ValueError("Parameter 'secret' is required in InferenceCalculator.")
        self.aux_cols = aux_cols
        self.secret = secret
        self.regression = regression
        self.n_attacks = n_attacks
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
        evaluator = InferenceEvaluator(
            ori=self.original_data,
            syn=self.synthetic_data,
            aux_cols=self.aux_cols,
            secret=self.secret,
            regression=self.regression,
            n_attacks=self.n_attacks,
            control=self.control
        )
        # Perform the evaluation and return the result
        return evaluator.evaluate().risk()


if __name__ == "__main__":
    original_data = pd.read_csv("/Users/ksi/Development/Bachelorthesis/insurance.csv")
    synthetic_data = pd.read_csv("/Users/ksi/Development/Bachelorthesis/SynPrivUtil_Framework/synprivutil/models/insurance_datasets/gmm_sample.csv")



    aux_cols = ["gender","height","weight"]
    secret = "age"

    aux_cols=["age",  "sex", "bmi"]
    secret='children'

    test_inference_calculator = InferenceCalculator(original_data, synthetic_data, aux_cols=aux_cols, secret=secret)
    test_dcr = test_inference_calculator.evaluate()
    print(f"INFERENCE {test_dcr}")

    synthetic_datasets = ["copulagan", "ctgan", "gaussian_copula", "gmm", "tvae", "random"]
    original_datasets =["diabetes"]
    print("STARTED")

    for orig in original_datasets:
        for syn in synthetic_datasets:
            original_data = pd.read_csv(f"/Users/ksi/Development/Bachelorthesis/SynPrivUtil_Framework/synprivutil/models/{orig}_datasets/train/{orig}.csv")
            synthetic_data = pd.read_csv(f"/Users/ksi/Development/Bachelorthesis/SynPrivUtil_Framework/synprivutil/models/{orig}_datasets/syn_on_train/{syn}_sample.csv")
            control_orig = pd.read_csv(f"/Users/ksi/Development/Bachelorthesis/SynPrivUtil_Framework/synprivutil/models/{orig}_datasets/test/{orig}.csv")
            test_sing = InferenceCalculator(original_data, synthetic_data, control=control_orig)

            t = test_sing.evaluate()
            print(f"SINGLING OUT; {orig}, {syn}")
            print(t)


