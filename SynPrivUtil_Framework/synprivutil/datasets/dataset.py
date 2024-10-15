from abc import abstractmethod, ABC


class Dataset(ABC):
    """
    Base class for the dataset object.

    """

    @abstractmethod
    def read(self, input_path):
        """
        Read dataset and description file.

        """
        pass

    @abstractmethod
    def write(self, output_path):
        """
        Write dataset and description to file.

        """
        pass

    @abstractmethod
    def read_from_string(self, data, schema):
        """
        Read from dataset and description as strings.
        """
        pass

    @abstractmethod
    def write_to_string(self):
        """
        Write dataset to a string.
        """
        pass

    @abstractmethod
    def sample(self, n_samples):
        """
        Sample from dataset a set of records.

        """
        pass

    @abstractmethod
    def get_records(self, record_ids):
        """
        Select and return a record(s).

        """
        pass

    @abstractmethod
    def drop_records(self, record_ids):
        """
        Drop a record(s) and return modified dataset.

        """
        pass

    @abstractmethod
    def add_records(self, records):
        """
        Add record(s) to dataset and return modified dataset.

        """
        pass

    @abstractmethod
    def replace(self, record_in, record_out):
        """
        Replace a row with a given row.

        """
        pass

    @abstractmethod
    def create_subsets(self, n, sample_size, drop_records=None):
        """
        Create a number of training datasets (sub-samples from main dataset)
        of a given sample size and with the option to remove some records.

        """
        pass

    @abstractmethod
    def __add__(self, other):
        """
        Adding two Dataset objects together.

        """
        pass

    @abstractmethod
    def __iter__(self):
        """
        Returns an iterator over records in the dataset.
        """
        pass

    @property
    def label(self):
        return "Unnamed dataset"