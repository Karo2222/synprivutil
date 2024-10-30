from abc import ABC, abstractmethod
import pandas as pd

from privacy_utility_framework.privacy_utility_framework.dataset.dataset import DatasetManager


# TODO: Normalize data
class PrivacyMetricCalculator(ABC):
    def __init__(self, original: pd.DataFrame, synthetic: pd.DataFrame, original_name: str = None, synthetic_name: str = None, **kwargs):
        # Check if the provided data is of type Dataset
        self.original = None
        self.synthetic = None
        if not isinstance(original, pd.DataFrame):
            raise TypeError("original_data must be an instance of pandas Dataframe.")
        if not isinstance(synthetic, pd.DataFrame):
            raise TypeError("synthetic_data must be an instance of pandas Dataframe.")
        self.transform_and_normalize(original, synthetic, original_name, synthetic_name)
        # NOTE: Can be removed if not used by any class
        self._handle_additional_inputs(**kwargs)
        self._validate_data()

    def _handle_additional_inputs(self, **kwargs):
        pass

    @abstractmethod
    def evaluate(self) -> float:
        pass

    def _validate_data(self):
        """
        Validates that the original and synthetic datasets have the same columns and compatible data types.

        :raises ValueError: If the datasets do not match in structure.
        """
        # Check column names
        if set(self.original.data.columns) != set(self.synthetic.data.columns):
            raise ValueError("Column names do not match between original and synthetic datasets.")

        # Check #columns
        if len(self.original.data.columns) != len(self.synthetic.data.columns):
            raise ValueError("Number of columns do not match between original and synthetic datasets.")

        #Check for missing values
        assert not self.original.data.isnull().any().any(), "Original dataset contains missing values."
        assert not self.synthetic.data.isnull().any().any(), "Synthetic dataset contains missing values."

        # Check that data types are compatible
        for col in self.original.data.columns:
            if self.original.data[col].dtype != self.synthetic.data[col].dtype:
                raise ValueError(f"Data type mismatch in column '{col}'.")
            pass

    def transform_and_normalize(self, original, synthetic, original_name, synthetic_name):
        manager = DatasetManager(original, synthetic, original_name, synthetic_name)

        # Set the transformer and scaler for the datasets
        manager.set_transformer_and_scaler_for_datasets()

        # Transform and normalize the datasets
        manager.transform_and_normalize_datasets()
        self.original = manager.original_dataset
        self.synthetic = manager.synthetic_dataset
