from typing import List, Tuple, Optional

import pandas as pd
from anonymeter.evaluators import LinkabilityEvaluator

from SynPrivUtil_Framework.synprivutil.privacy_metrics import PrivacyMetricCalculator

# TODO: extend params


class LinkabilityCalculator(PrivacyMetricCalculator):
    def __init__(self, original_data: pd.DataFrame, synthetic_data: pd.DataFrame,
                 aux_cols: Tuple[List[str], List[str]],
                 n_attacks: Optional[int] = 1000,
                 n_neighbors: int = 5,
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
        super().__init__(original_data, synthetic_data, aux_cols=aux_cols)
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
            ori=self.original_data,
            syn=self.synthetic_data,
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
