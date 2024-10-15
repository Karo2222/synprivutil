import pandas as pd
from rdt import HyperTransformer
from rdt.transformers import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def transform_and_normalize(original_data, synthetic_data=None):
    transformed_orig, transformed_syn, transformer = transform_rdt(original_data, synthetic_data)

    norm_orig, norm_syn = normalize(transformed_orig, transformed_syn)
    # # Initialize the MinMaxScaler
    # scaler = MinMaxScaler()
    # orig_norm = scaler.fit_transform(transformed_orig)
    # orig_scaled = pd.DataFrame(orig_norm, columns=transformed_orig.columns)
    # syn_scaled = None
    # if transformed_syn is not None:
    #     syn_norm = scaler.transform(transformed_syn)
    #     syn_scaled = pd.DataFrame(syn_norm, columns=transformed_syn.columns)
    #     #print("Synthetic Data was also transformed.")
    return norm_orig, norm_syn


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


def dynamic_train_test_split(df: pd.DataFrame, small_threshold: int = 1000, large_threshold: int = 50000):
    """
    Split the DataFrame into training and test sets with a dynamic test size based on the DataFrame size.

    Parameters:
    - df: pd.DataFrame
      The DataFrame to split.
    - small_threshold: int (default: 1000)
      The threshold below which a smaller test size is used.
    - large_threshold: int (default: 50000)
      The threshold above which a larger test size is used.

    Returns:
    - train_df: pd.DataFrame
      The training set.
    - test_df: pd.DataFrame
      The test set.
    """
    num_rows = len(df)

    if num_rows <= small_threshold:
        test_size = 0.1  # 10% for very small datasets
    elif num_rows <= large_threshold:
        test_size = 0.2  # 20% for medium-sized datasets
    else:
        test_size = 0.3  # 30% for large datasets
    print(f"Test size was: {test_size}")
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    return train_df, test_df
