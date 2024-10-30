import pandas as pd
from rdt import HyperTransformer
from rdt.transformers import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler



def normalize(original_data, synthetic_data):
    non_categorical_cols = original_data.select_dtypes(include=[float, int]).columns
    scaler = MinMaxScaler()
    orig_norm = scaler.fit_transform(original_data[non_categorical_cols])
    orig_norm_df = pd.DataFrame(orig_norm, columns=non_categorical_cols, index=original_data.index)
    orig_scaled = original_data.copy()
    orig_scaled[non_categorical_cols] = orig_norm_df
    syn_scaled = None
    if synthetic_data is not None:
        syn_norm = scaler.transform(synthetic_data[non_categorical_cols])
        syn_norm_df = pd.DataFrame(syn_norm, columns=non_categorical_cols, index=synthetic_data.index)
        syn_scaled = synthetic_data.copy()
        syn_scaled[non_categorical_cols] = syn_norm_df
    return orig_scaled, syn_scaled


def transform_rdt(original_data, synthetic_data=None):
    categorical_columns = original_data.select_dtypes(include=['object', 'category']).columns
    # print(categorical_columns)

    ht = HyperTransformer()
    ht.detect_initial_config(data=original_data)
    existing_config = ht.get_config()
    # Change default FloatFormatter to OneHotEncoder for categorical columns
    transformers = {col: OneHotEncoder() for col in categorical_columns}
    existing_config['transformers'].update(transformers)
    # print("~~~~~~~~~CONFIG~~~~~~~~~")
    # print(ht.get_config())
    ht.fit(original_data)
    transformed_orig = ht.transform(original_data)
    transformed_syn = None
    if synthetic_data is not None:
        transformed_syn = ht.transform(synthetic_data)
        # print("Synthetic Data was also transformed.")

    return transformed_orig, transformed_syn, ht



