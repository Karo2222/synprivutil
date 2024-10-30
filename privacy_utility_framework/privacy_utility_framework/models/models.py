from abc import abstractmethod, ABC

import numpy as np
import pandas as pd
from rdt import HyperTransformer
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer, CopulaGANSynthesizer, TVAESynthesizer
from sdv.metadata import SingleTableMetadata
from sklearn.mixture import GaussianMixture
from privacy_utility_framework.privacy_utility_framework.models.transform import transform_and_normalize, transform_rdt, \
    dynamic_train_test_split
from privacy_utility_framework.privacy_utility_framework.synthesizers.synthesizers import GaussianMixtureModel, \
    GaussianCopulaModel, CTGANModel, CopulaGANModel, TVAEModel, RandomModel

orig = "diabetes"
folder = f"{orig}_datasets"
make_train = False
use_train = False
generate_new_syn = False
if use_train:
    data = pd.read_csv(
        f'/privacy_utility_framework/synprivutil/models/{folder}/train/{orig}.csv',
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




if generate_new_syn:

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)
    gmm_model = GaussianMixtureModel(max_components=10)
    gmm_model.fit(data)
    gmm_model.save_sample(f"{folder}/gmm_sample.csv", len(data))


    # Instantiate and use the models
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
