import pandas as pd


class TabularDataset:
    def __init__(self, data):
        assert isinstance(data, pd.DataFrame), 'Data needs to be a pandas Dataframe'
        self.data = data

    @classmethod
    def read(cls, filepath):
        pass

    def write(self, filepath):
        pass

    def sample(self, n_samples=1, frac=None, random_state=None):
        """
        Sample a set of records from a TabularDataset object.

        Parameters
        ----------
        n_samples : int
            Number of records to sample. If frac is not None, this parameter is ignored.

        frac : float
            Fraction of records to sample.

        random_state : optional
            Passed to `pandas.DataFrame.sample()`

        Returns
        -------
        TabularDataset
            A TabularDataset object with a sample of the records of the original object.

        """
        if frac:
            n_samples = int(frac * len(self))

        return TabularDataset(
            data=self.data.sample(n_samples, random_state=random_state)
        )

    def get_records(self, record_ids):
        """
        Get a record from the TabularDataset object

        Parameters
        ----------
        record_ids : list[int]
            List of indexes of records to retrieve.

        Returns
        -------
        TabularDataset
            A TabularDataset object with the record(s).

        """

        # TODO: what if the index is supposed to be a column? an identifier?
        if len(record_ids) == 1:
            return TabularRecord(
                self.data.iloc[record_ids], self.description, record_ids[0]
            )
        return TabularDataset(self.data.iloc[record_ids], self.description)


    def create_subsets(self, n, sample_size, drop_records=False):
        """
        Create a number n of subsets of this dataset of size sample_size without
        replacement. If needed, the records can be dropped from this dataset.

        Parameters
        ----------
        n : int
            Number of datasets to create.
        sample_size : int
            Size of the subset datasets to be created.
        drop_records: bool
            Whether to remove the records sampled from this dataset (in place).

        Returns
        -------
        list(TabularDataset)
            A lists containing subsets of the data with and without the target record(s).

        """
        assert sample_size <= len(
            self
        ), f"Cannot create subsets larger than original dataset, sample_size max: {len(self)} got {sample_size}"

        # Create splits.
        splits = index_split(self.data.shape[0], sample_size, n)

        # Returns a list of TabularDataset subsampled from this dataset.
        subsamples = [self.get_records(train_index) for train_index in splits]

        # If required, remove the indices from the dataset.
        if drop_records:
            for train_index in splits:
                self.drop_records(train_index, in_place=True)

        return subsamples



    def copy(self):
        """
        Create a TabularDataset that is a deep copy of this one. In particular,
        the underlying data is copied and can thus be modified freely.

        Returns
        -------
        TabularDataset
            A copy of this TabularDataset.

        """
        return TabularDataset(self.data.copy(), self.description)

    def view(self, columns = None, exclude_columns = None):
        """
        Create a TabularDataset object that contains a subset of the columns of
        this TabularDataset. The resulting object only has a copy of the data,
        and can thus be modified without affecting the original data.

        Parameters
        ----------
        Exactly one of `columns` and `exclude_columns` must be defined.

        columns: list, or None
            The columns to include in the view.
        exclude_columns: list, or None
            The columns to exclude from the view, with all other columns included.

        Returns
        -------
        TabularDataset
            A subset of this data, restricted to some columns.

        """
        assert (
                columns is not None or exclude_columns is not None
        ), "Empty view: specify either columns or exclude_columns."
        assert (
                columns is None or exclude_columns is None
        ), "Overspecified view: only one of columns and exclude_columns can be given."

        if exclude_columns is not None:
            columns = [c for c in self.description.columns if c not in exclude_columns]

        return TabularDataset(self.data[columns], self.description.view(columns))

    @property
    def transform_and_normalize(self):
        """
        Encodes this dataset as a np.array, where numeric values are kept as is
        and categorical values are 1-hot encoded. This is only computed once
        (for efficiency reasons), so beware of modifying TabularDataset after
        using this property.

        The columns are kept in the order of the description, with categorical
        variables encoded over several contiguous columns.

        Returns
        -------
        np.array

        """
        pass



    def __len__(self):
        """
        Returns the number of records in this dataset.

        Returns
        -------
        integer
            length: number of records in this dataset.

        """
        return self.data.shape[0]

    def __contains__(self, item):
        """
        Determines the truth value of `item in self`. The only items considered
        to be in a TabularDataset are the rows, treated as 1-row TabularDatasets.

        Parameters
        ----------
        item : Object
            Object to check membership of.

        Returns
        -------
        bool
            Whether or not item is considered to be contained in self.

        """
        if not isinstance(item, TabularDataset):
            raise ValueError(
                f"Only TabularDatasets can be checked for containment, not {type(item)}"
            )
        if len(item) != 1:
            raise ValueError(
                f"Only length-1 TabularDatasets can be checked for containment, got length {len(item)})"
            )

        return (self.data == item.data.iloc[0]).all(axis=1).any()

    @property
    def label(self):
        return self.description.label

