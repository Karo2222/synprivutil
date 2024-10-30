import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.mixture import GaussianMixture
import numpy as np
import joblib


# NOTE: I am testing stuff with the different datsets in here

def read_and_transform_data(file_path):
    """
    Read data from a CSV file and transform it for analysis.
    
    Parameters:
    file_path (str): The path to the CSV file containing the data.
    
    Returns:
    pd.DataFrame: Transformed data with categorical columns one-hot encoded and numerical columns normalized.
    """
    # Read in the data
    df = pd.read_csv(file_path)
    df_clean = df.dropna()
    # Separate categorical and numerical columns
    categorical_columns = df_clean.select_dtypes(include=['object']).columns.tolist()
    numerical_columns = df_clean.select_dtypes(include=['number']).columns.tolist()

    print(categorical_columns)

    # Initialize encoder and scaler to None
    encoder = None
    scaler = None
    
    # One-hot encode categorical columns if they exist
    if categorical_columns:
        encoder = OneHotEncoder(sparse_output=False)
        categorical_data = encoder.fit_transform(df_clean[categorical_columns])
        categorical_df = pd.DataFrame(categorical_data, columns=encoder.get_feature_names_out(categorical_columns))
    else:
        categorical_df = pd.DataFrame()

    # TODO: check negative numbers in synthetic_data_transformed
    # Normalize numerical columns if they exist
    if numerical_columns:
        scaler = MinMaxScaler()
        numerical_data = scaler.fit_transform(df_clean[numerical_columns])
        numerical_df = pd.DataFrame(numerical_data, columns=numerical_columns)
    else:
        numerical_df = pd.DataFrame()

    # Combine transformed data into a single DataFrame
    transformed_df = pd.concat([categorical_df, numerical_df], axis=1)
    return transformed_df, encoder, scaler

def generate_synthetic_data(real_data, n_samples, n_components=10):
    """
    Generate synthetic data using a Gaussian Mixture Model.

    OR/AND

    Generate synthetic data using a CTGAN Model.
    
    Parameters:
    real_data (pd.DataFrame): The original dataset.
    n_samples (int): The number of synthetic samples to generate.
    n_components (int): The number of mixture components for the Gaussian Mixture Model.
    
    Returns:
    pd.DataFrame: The generated synthetic data.
    """
    # Fit a Gaussian Mixture Model to the real data
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(real_data)

    #metadata = SingleTableMetadata()
    #metadata.detect_from_dataframe(real_data)
    #model = CTGANSynthesizer(metadata=metadata)
    #model.fit(real_data)
    #synthetic_df = model.sample(num_rows=n_samples)
    
    # Generate synthetic data
    synthetic_data = gmm.sample(n_samples)[0]
    synthetic_df = pd.DataFrame(synthetic_data, columns=real_data.columns)
    return synthetic_df

if __name__ == "__main__":
    # Paths to the real data file
    real_data_path = '/Users/ksi/Development/Bachelorthesis/SD2011_selected_columns.csv'
    
    # Read and transform the real data
    real_data, encoder, scaler = read_and_transform_data(real_data_path)
    
    # Generate synthetic data
    synthetic_data = generate_synthetic_data(real_data, len(real_data))

    # Save transformed original data to a CSV file
    real_data.to_csv('/Users/ksi/Development/Bachelorthesis/self_transformed_SD2011_selected_columns.csv', index=False)
    # Save transformed synthetic data to a CSV file
    synthetic_data.to_csv('/Users/ksi/Development/Bachelorthesis/self_transformed_syn_SD2011_selected_columns.csv', index=False)
    
    # Inverse transform synthetic data to original form
    if encoder is not None:
        # Find out the number of categorical columns
        n_categorical_cols = len(encoder.get_feature_names_out())
        
        # Separate the synthetic data into categorical and numerical parts
        categorical_data = synthetic_data.iloc[:, :n_categorical_cols]
        numerical_data = synthetic_data.iloc[:, n_categorical_cols:]

        # Inverse transform categorical data
        categorical_cols = encoder.inverse_transform(categorical_data)
        categorical_df = pd.DataFrame(categorical_cols, columns=real_data.select_dtypes(include=['object']).columns)
        
        # Combine categorical and numerical data
        synthetic_data = pd.concat([categorical_df, numerical_data], axis=1)
    else:
        synthetic_data = pd.DataFrame(synthetic_data, columns=real_data.columns)
    
    if scaler is not None:
        # Find out the number of numerical columns
        numerical_cols = real_data.select_dtypes(include=['number']).columns.tolist()
        
        # Inverse transform numerical data
        numerical_data = scaler.inverse_transform(synthetic_data[numerical_cols])
        numerical_df = pd.DataFrame(numerical_data, columns=numerical_cols)
        synthetic_data[numerical_cols] = numerical_df
    
    # Save synthetic data to a CSV file
    synthetic_data.to_csv('/Users/ksi/Development/Bachelorthesis/self_syn_SD2011_selected_columns.csv', index=False)
    print(f"Synthetic data generated and saved to path/to/synthetic_data.csv")
    if encoder is not None:
        joblib.dump(encoder, 'encoder.pkl')
        print(f"Encoder saved: {type(encoder).__name__}")

    if scaler is not None:
        joblib.dump(scaler, 'scaler.pkl')
        print(f"Scaler saved: {type(scaler).__name__}")

