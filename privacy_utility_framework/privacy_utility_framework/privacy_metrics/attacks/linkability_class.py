from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from anonymeter.evaluators import LinkabilityEvaluator

from privacy_utility_framework.privacy_utility_framework.privacy_metrics import PrivacyMetricCalculator

# TODO: extend params


class LinkabilityCalculator(PrivacyMetricCalculator):
    def __init__(self, original: pd.DataFrame, synthetic: pd.DataFrame,
                 aux_cols: Tuple[List[str], List[str]],
                 n_attacks: Optional[int] = 2000,
                 n_neighbors: int = 10,
                 control: Optional[pd.DataFrame] = None):
        """
       Parameters:
       - ori: pd.DataFrame
         The original dataset.
       - syn: pd.DataFrame
         The synthetic dataset.
       - aux_cols: Tuple[List[str], List[str]]
         Tuple containing two lists of auxiliary columns.
       - n_attacks: Optional[int] (default: 500)
         Number of attacks.
       - n_neighbors: int (default: 1)
         Number of neighbors.
       - control: Optional[pd.DataFrame] (default: None)
         Control dataset.
       """
        super().__init__(original, synthetic, aux_cols=aux_cols)
        if aux_cols is None:
            raise ValueError("Parameter 'aux_cols' is required in LinkabilityCalculator.")
        self.aux_cols = aux_cols
        self.n_attacks = n_attacks
        self.n_neighbors = n_neighbors
        self.control = control

    # def _prepare_evaluator_args(self):
    #     # Prepare the arguments for LinkabilityEvaluator
    #     args = {
    #         'ori': self.original_data,
    #         'syn': self.synthetic_data
    #     }
    #
    #     # Only include aux_cols if it's not None
    #     if self.aux_cols is not None:
    #         args['aux_cols'] = self.aux_cols
    #
    #     return args
    #
    # def evaluate(self):
    #     # Prepare arguments dynamically
    #     evaluator_args = self._prepare_evaluator_args()
    #
    #     evaluator = LinkabilityEvaluator(**evaluator_args)
    #
    #     return evaluator.evaluate().risk()

    def evaluate(self):
        evaluator = LinkabilityEvaluator(
            ori=self.original,
            syn=self.synthetic,
            aux_cols=self.aux_cols,
            n_attacks=self.n_attacks,
            n_neighbors=self.n_neighbors,
            control=self.control
        )
        return evaluator.evaluate().risk()
# orig_sample = self.original_data.sample(n=n_samples, random_state=np.random.randint(0, 10000))
# syn_sample = self.synthetic_data.sample(n=n_samples, random_state=np.random.randint(0, 10000))

# original_data = pd.read_csv("/Users/ksi/Development/Bachelorthesis/insurance.csv")
# synthetic_data = pd.read_csv("/Users/ksi/Development/Bachelorthesis/SynPrivUtil_Framework/synprivutil/models/insurance_datasets/ctgan_sample.csv")
#
#
# aux_cols=["age",  "sex", "bmi"]
# aux_cols = (['age', 'sex', 'bmi', 'children'], ['smoker', 'region', 'charges'])
# test_inference_calculator = LinkabilityCalculator(original_data, synthetic_data, aux_cols=aux_cols)
# test_dcr = test_inference_calculator.evaluate()
# print(f"LINKABILITY {test_dcr}")

if __name__ == "__main__":
    synthetic_datasets = ["copulagan", "ctgan", "gaussian_copula", "gmm", "tvae", "random"]
    original_datasets =["insurance"]
    aux_cols_d = (['Age','DiabetesPedigreeFunction'], ['BMI', 'Glucose', 'BloodPressure', "Pregnancies"])
    aux_cols_c = (["age", "gender", "height", "weight"], [ "ap_hi","ap_lo",  "cardio", "cholesterol", "gluc"])
    aux_cols_i = (["age", "sex", "bmi"], ["children", "smoker", "region", "charges"])

    print("STARTED")

    for orig in original_datasets:
        for syn in synthetic_datasets:
            original_data = pd.read_csv(
                f"/privacy_utility_framework/synprivutil/models/{orig}_datasets/train/{orig}.csv")
            synthetic_data = pd.read_csv(
                f"/privacy_utility_framework/synprivutil/models/{orig}_datasets/syn_on_train/{syn}_sample.csv")
            # original_data = pd.read_csv(f"/Users/ksi/Development/Bachelorthesis/{orig}.csv")
            # synthetic_data = pd.read_csv(f"/Users/ksi/Development/Bachelorthesis/SynPrivUtil_Framework/synprivutil/models/{orig}_datasets/{syn}_sample.csv")

            control_orig = pd.read_csv(f"/privacy_utility_framework/synprivutil/models/{orig}_datasets/test/{orig}.csv")
            # orig_sample = original_data.sample(n=900, random_state=np.random.randint(0, 10000))
            # syn_sample = synthetic_data.sample(n=900, random_state=np.random.randint(0, 10000))
            test_sing = LinkabilityCalculator(original_data, synthetic_data, aux_cols=aux_cols_i, control=control_orig, n_attacks=260)

            t = test_sing.evaluate()
            print(f"LINK; {orig}, {syn}")
            print(t)
