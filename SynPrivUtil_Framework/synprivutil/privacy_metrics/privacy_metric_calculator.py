from abc import abstractmethod
from typing import Type, Any
import pandas as pd


# TODO: Normalize data
class PrivacyMetricCalculator:
    def __init__(self, original_data: pd.DataFrame, synthetic_data: pd.DataFrame, **kwargs):
        self.original_data = original_data
        self.synthetic_data = synthetic_data
        # NOTE: Can be removed if not used by any class
        self._handle_additional_inputs(**kwargs)
        self._validate_data()

    def _handle_additional_inputs(self, **kwargs):
        pass

    def evaluate(self) -> float:
        raise NotImplementedError("Subclasses should implement this method.")

    def _validate_data(self):
        """
        Validates that the original and synthetic datasets have the same columns and compatible data types.

        :raises ValueError: If the datasets do not match in structure.
        """
        # Check column names
        if set(self.original_data.columns) != set(self.synthetic_data.columns):
            raise ValueError("Column names do not match between original and synthetic datasets.")

        # Check #columns
        if len(self.original_data.columns) != len(self.synthetic_data.columns):
            raise ValueError("Number of columns do not match between original and synthetic datasets.")

        #Check for missing values
        assert not self.original_data.isnull().any().any(), "Original dataset contains missing values."
        assert not self.synthetic_data.isnull().any().any(), "Synthetic dataset contains missing values."

        # Check that data types are compatible
        for col in self.original_data.columns:
            if self.original_data[col].dtype != self.synthetic_data[col].dtype:
                raise ValueError(f"Data type mismatch in column '{col}'.")
            pass
