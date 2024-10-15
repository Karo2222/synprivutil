from typing import Optional

import numpy as np
import pandas as pd
from anonymeter.evaluators import InferenceEvaluator, SinglingOutEvaluator

from SynPrivUtil_Framework.synprivutil.privacy_metrics import PrivacyMetricCalculator


class SinglingOutCalculator(PrivacyMetricCalculator):
    def __init__(self, original_data: pd.DataFrame, synthetic_data: pd.DataFrame,
                 n_cols: int = 3,
                 max_attempts: Optional[int] = 10000000,
                 n_attacks: int = 500,
                 control: Optional[pd.DataFrame] = None):
        super().__init__(original_data, synthetic_data)
        self.n_cols = n_cols
        self.n_attacks = n_attacks
        self.control = control
        self.max_attempts = max_attempts
        #print(f"Max at {max_attempts}")

    def evaluate(self):
        evaluator = SinglingOutEvaluator(
            ori=self.original_data,
            syn=self.synthetic_data,
            n_attacks=self.n_attacks,
            n_cols=self.n_cols,
            control=self.control,
            max_attempts=self.max_attempts,
        )
        return evaluator.evaluate().risk()


if __name__ == "__main__":
    synthetic_datasets = ["copulagan", "ctgan", "gaussian_copula", "gmm", "tvae", "random"]
    original_datasets =["insurance"]
    print("STARTED")

    for orig in original_datasets:
        for syn in synthetic_datasets:
            original_data = pd.read_csv(f"/Users/ksi/Development/Bachelorthesis/SynPrivUtil_Framework/synprivutil/models/{orig}_datasets/train/{orig}.csv")
            synthetic_data = pd.read_csv(f"/Users/ksi/Development/Bachelorthesis/SynPrivUtil_Framework/synprivutil/models/{orig}_datasets/syn_on_train/{syn}_sample.csv")
            control_orig = pd.read_csv(f"/Users/ksi/Development/Bachelorthesis/SynPrivUtil_Framework/synprivutil/models/{orig}_datasets/test/{orig}.csv")
            orig_sample = original_data.sample(n=900, random_state=np.random.randint(0, 10000))
            syn_sample = synthetic_data.sample(n=900, random_state=np.random.randint(0, 10000))

            test_sing = SinglingOutCalculator(orig_sample, syn_sample, control=control_orig)

            t = test_sing.evaluate()
            print(f"SINGLING OUT; {orig}, {syn}")
            print(t)
