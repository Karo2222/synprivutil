from typing import List

import pandas as pd
from anonymeter.evaluators import LinkabilityEvaluator

from SynPrivUtil_Framework.synprivutil.privacy_metrics import PrivacyMetricCalculator

# TODO: extend params

class LinkabilityCalculator(PrivacyMetricCalculator):
    def __init__(self, original_data: pd.DataFrame, synthetic_data: pd.DataFrame, aux_cols: tuple[list[str], list[str]]):
        super().__init__(original_data, synthetic_data, aux_cols=aux_cols)
        if aux_cols is None:
            raise ValueError("Parameter 'aux_cols' is required in LinkabilityCalculator.")
        self.aux_cols = aux_cols

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
            aux_cols=self.aux_cols
        )
        return evaluator.evaluate().risk()
