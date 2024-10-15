from abc import abstractmethod, ABC

import numpy as np
import pandas as pd
from rdt import HyperTransformer
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer, CopulaGANSynthesizer, TVAESynthesizer
from sdv.metadata import SingleTableMetadata
from sklearn.mixture import GaussianMixture
from SynPrivUtil_Framework.synprivutil.models.transform import transform_and_normalize, transform_rdt, \
    dynamic_train_test_split

orig = "diabetes"
folder = f"{orig}_datasets"
make_train = False
use_train = False
generate_new_syn = False
if use_train:
    data = pd.read_csv(
        f'/Users/ksi/Development/Bachelorthesis/SynPrivUtil_Framework/synprivutil/models/{folder}/train/{orig}.csv',
        delimiter=',')
    folder = f"{folder}/syn_on_train"

else:
    data = pd.read_csv(f'/Users/ksi/Development/Bachelorthesis/{orig}.csv', delimiter=',')
if make_train and not use_train:
    train, test = dynamic_train_test_split(data)
    print(f"train {train}")
    train.to_csv(f"{folder}/train/{orig}.csv", index=False)
    test.to_csv(f"{folder}/test/{orig}.csv", index=False)


# # Update a column's metadata
# metadata.update_column(
#     column_name='Open',
#     sdtype='numerical',  # or 'numerical', 'datetime', categorical etc.
# )
#
# metadata.update_column(
#     column_name='High',
#     sdtype='numerical',  # or 'numerical', 'datetime', categorical etc.
# )
#
# metadata.update_column(
#     column_name='Low',
#     sdtype='numerical',  # or 'numerical', 'datetime', categorical etc.
# )
#
# metadata.update_column(
#     column_name='Volume',
#     sdtype='numerical',  # or 'numerical', 'datetime', categorical etc.
# )
# metadata.update_column(
#     column_name='Close',
#     sdtype='numerical',  # or 'numerical', 'datetime', categorical etc.
# )

# # Validate metadata
# metadata.validate_data(data)
#
# # Save metadata to a JSON file
# # metadata.save_to_json('my_final_metadata.json')
# print("~~~~~~~~~METADATA~~~~~~~~~")
# print(metadata)
#
# print("~~~~~TRANSFORMED DATA~~~~~")


class BaseModel(ABC):
    synthesizer_class = None

    def __init__(self, synthesizer):
        self.synthesizer = synthesizer

    def fit(self, data):
        self.synthesizer.fit(data)

    def sample(self, num_samples=200):
        return self.synthesizer.sample(num_samples)

    def save_sample(self, filename, num_samples=200):
        synthetic_data = self.sample(num_samples)
        synthetic_data.to_csv(filename, index=False)
        print(f"Data saved to {filename}")

    def save_model(self, filename):
        self.synthesizer.save(filename)
        print(f"Model saved to {filename}")

    @classmethod
    def load_model(cls, filepath):
        # Load the synthesizer using the CTGANSynthesizer load method
        synthesizer = cls.synthesizer_class.load(filepath)
        # Create an instance of CTGANModel using the loaded synthesizer
        instance = cls.__new__(cls)  # Create an instance without calling __init__
        instance.synthesizer = synthesizer  # Manually set the synthesizer
        return instance


class GaussianCopulaModel(BaseModel):
    synthesizer_class = GaussianCopulaSynthesizer

    def __init__(self, metadata):
        super().__init__(GaussianCopulaSynthesizer(metadata))


class CTGANModel(BaseModel):
    synthesizer_class = CTGANSynthesizer

    def __init__(self, metadata):
        super().__init__(CTGANSynthesizer(metadata))

    # @classmethod
    # def load_model(cls, filepath):
    #     # Load the synthesizer using the CTGANSynthesizer load method
    #     synthesizer = CTGANSynthesizer.load(filepath)
    #     # Create an instance of CTGANModel using the loaded synthesizer
    #     instance = cls.__new__(cls)  # Create an instance without calling __init__
    #     instance.synthesizer = synthesizer  # Manually set the synthesizer
    #     return instance


class CopulaGANModel(BaseModel):
    synthesizer_class = CopulaGANSynthesizer

    def __init__(self, metadata):
        super().__init__(CopulaGANSynthesizer(metadata))


class TVAEModel(BaseModel):
    synthesizer_class = TVAESynthesizer

    def __init__(self, metadata):
        super().__init__(TVAESynthesizer(metadata))


class GaussianMixtureModel(BaseModel):
    def __init__(self, max_components=10):
        super().__init__(None)
        self.transformed_data = None
        self.transformer = None
        self.max_components = max_components
        self.model = None

    def fit(self, data, random_state=42):
        self.transformed_data, _, self.transformer = transform_rdt(data)
        # Select the optimal number of components
        optimal_n_components = self._select_n_components(self.transformed_data, random_state)
        self.model = GaussianMixture(n_components=optimal_n_components, random_state=random_state)
        self.model.fit(self.transformed_data)

    def sample(self, num_samples=200):
        if self.model is not None:
            samples, _ = self.model.sample(num_samples)
            samples_pd = pd.DataFrame(samples, columns=self.transformed_data.columns)
            inverse_samples = self.transformer.reverse_transform(samples_pd)
            return inverse_samples
        else:
            raise RuntimeError("Data has not been fitted yet.")

    def save_sample(self, filename, num_samples=200):
        samples = self.sample(num_samples)
        # Inverse transform to original format
        samples.to_csv(filename, index=False)
        print(f"Data saved to {filename}")

    def save_model(self, filename):
        pass
        # np.save(filename + '_weights', self.model.weights_, allow_pickle=False)
        # np.save(filename + '_means', self.model.means_, allow_pickle=False)
        # np.save(filename + '_covariances', self.model.covariances_, allow_pickle=False)
        # np.save(filename + '_precisions_cholesky', self.model.precisions_cholesky_, allow_pickle=False)
        # print(f"Model saved to {filename}")

    @classmethod
    def load_model(cls, filepath):
        pass
        # means = np.load(filepath + '_means.npy')
        # covar = np.load(filepath + '_covariances.npy')
        # loaded_gmm = GaussianMixture(n_components=len(means), covariance_type='full')
        # loaded_gmm.precisions_cholesky_ = np.load(filepath + '_precisions_cholesky.npy')
        # loaded_gmm.weights_ = np.load(filepath + '_weights.npy')
        # loaded_gmm.means_ = means
        # loaded_gmm.covariances_ = covar
        # instance = cls.__new__(cls)  # Create an instance without calling __init__
        # instance.model = loaded_gmm
        # print("Gaussian Mixture Model was loaded.")
        # return instance

    def _select_n_components(self, data, random_state):
        """
        Select the optimal number of components for GMM using BIC.

        Parameters:
        data: The dataset to fit the GMM.
        max_components: The maximum number of components to try.

        Returns:
        The optimal number of components.
            """
        bics = []
        n_components_range = range(1, self.max_components + 1)

        for n in n_components_range:
            gmm = GaussianMixture(n_components=n, random_state=random_state)
            gmm.fit(data)
            bics.append(gmm.bic(data))

        optimal_n_components = n_components_range[np.argmin(bics)]
        return optimal_n_components


class RandomModel(BaseModel):
    def __init__(self):
        super().__init__(None)
        self.data = None
        self.trained = False

    def fit(self, data):
        self.trained = True
        self.data = data  # No fitting needed for random sampling, simply set dataset

    def sample(self, num_samples=None, random_state=None):
        if self.trained:
            if num_samples is None:
                return self.data
            return pd.DataFrame(self.data.sample(num_samples, random_state=random_state, replace=False))
        else:
            raise RuntimeError("No dataset provided to generator")

    def save_model(self, filename):
        pass

    @classmethod
    def load_model(cls, filepath):
        pass

    def __call__(self, dataset, num_samples, random_state=None):
        self.fit(dataset)
        return self.sample(num_samples, random_state=random_state)

if generate_new_syn:

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)
    gmm_model = GaussianMixtureModel(max_components=10)
    gmm_model.fit(data)
    gmm_model.save_sample(f"{folder}/gmm_sample.csv", len(data))


# # # Instantiate and use the models
    gaussian_copula_model = GaussianCopulaModel(metadata)
    gaussian_copula_model.fit(data)
    gaussian_copula_model.save_sample(f"{folder}/gaussian_copula_sample.csv", len(data))
    gaussian_copula_model.save_model(f"{folder}/gaussian_copula_model.pkl")

    # #
    # ctgan_model = CTGANModel.load_model(
    #     "/Users/ksi/Development/Bachelorthesis/SynPrivUtil_Framework/synprivutil/models/diabetes_datasets/ctgan_model.pkl")
    # r = ctgan_model.sample(200)
    # print(r)
    #
    # gaussian_model = GaussianCopulaModel.load_model(
    #     "/Users/ksi/Development/Bachelorthesis/SynPrivUtil_Framework/synprivutil/models/diabetes_datasets/gaussian_copula_model.pkl")
    # t = gaussian_model.sample(200)
    # print(t)

    ctgan_model = CTGANModel(metadata)
    ctgan_model.fit(data)
    ctgan_model.save_sample(folder + "/" + "ctgan_sample.csv", len(data))
    ctgan_model.save_model(f"{folder}/ctgan_model.pkl")

    #
    copulagan_model = CopulaGANModel(metadata)
    copulagan_model.fit(data)
    copulagan_model.save_sample(f"{folder}/copulagan_sample.csv", len(data))
    copulagan_model.save_model(f"{folder}/copulagan_model.pkl")

    tvae_model = TVAEModel(metadata)
    tvae_model.fit(data)
    tvae_model.save_sample(f"{folder}/tvae_sample.csv", len(data))
    tvae_model.save_model(f"{folder}/tvae_model.pkl")

    random_vanilla_model = RandomModel()
    random_vanilla_model.fit(data)
    random_vanilla_model.save_sample(f"{folder}/random_sample.csv", len(data))
