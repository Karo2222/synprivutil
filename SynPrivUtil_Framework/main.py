import numpy as np
import pandas as pd
import joblib
import os

from scipy.spatial.distance import cdist

from SynPrivUtil_Framework.synprivutil.privacy_metrics.adversarial_accuracy_class import AdversarialAccuracyCalculator
from SynPrivUtil_Framework.synprivutil.utility_metrics.wasserstein import WassersteinCalculator
from synprivutil.privacy_metrics import *
from synprivutil.preprocessing import *

# NOTE: I am testing stuff with the different datsets in here

def main():

    # Version with data from synthpop package
    #real_data = pd.read_csv('/Users/ksi/Development/Bachelorthesis/transformed_SD2011_selected_columns.csv')
    #synthetic_data = pd.read_csv('/Users/ksi/Development/Bachelorthesis/transformed_syn_SD2011_selected_columns.csv')

    # Version with Gaussian Mixture
    real_data = pd.read_csv('/Users/ksi/Development/Bachelorthesis/diabetes_transformed.csv')
    synthetic_data = pd.read_csv('/Users/ksi/Development/Bachelorthesis/synthetic_data_transformed.csv')

    # Version with CTGAN
    synthetic_data = pd.read_csv('/Users/ksi/Development/Bachelorthesis/synthetic_data_transformed_ctgan.csv')

    #real_data = pd.read_csv('/Users/ksi/Development/Bachelorthesis/self_transformed_SD2011_selected_columns.csv')
    #synthetic_data = pd.read_csv('/Users/ksi/Development/Bachelorthesis/self_transformed_syn_SD2011_selected_columns.csv')

# test_path = '/Users/ksi/Development/Bachelorthesis/test.csv'
    #
    # # Read and transform the real data
    # real_data, encoder, scaler = read_and_transform_data(test_path)
    #
    # # Save transformed original data to a CSV file
    # real_data.to_csv('/Users/ksi/Development/Bachelorthesis/test_transformed.csv', index=False)

    p = PrivacyMetricManager(real_data, synthetic_data)

    p.add_metric(AdversarialAccuracyCalculator)
    p.add_metric(NNDRCalculator)
    p.add_metric(DCRCalculator, distance_metric='cityblock')

    # Version compatible with diabetes dataset
    p.add_metric(LinkabilityCalculator, aux_cols=["Glucose","BloodPressure","SkinThickness"])
    p.add_metric(InferenceCalculator, aux_cols=["Glucose","BloodPressure","SkinThickness"], secret='Outcome')
    p.add_metric(SinglingOutCalculator, aux_cols=["Glucose","BloodPressure","SkinThickness"], secret='Outcome')

    # Version compatible with data from synthpop package
    # p.add_metric(LinkabilityCalculator, aux_cols=["sex", "age", "region", "placesize"])
    # p.add_metric(InferenceCalculator, aux_cols=["sex", "age", "region", "placesize"], secret='workab')
    # p.add_metric(SinglingOutCalculator, aux_cols=["sex", "age", "region", "placesize"], secret='workab')

    t = WassersteinCalculator(real_data, synthetic_data)
    print(t.evaluate())
    # Evaluate all added metrics
    results = p.evaluate_all()
    for key, value in results.items():
        print(f"{key}: {value}")
    return

    # # Initialize encoder and scaler
    # encoder = None
    # scaler = None
    #
    # # Check and load the encoder if it exists
    # if os.path.exists('encoder.pkl'):
    #     encoder = joblib.load('encoder.pkl')
    # else:
    #     print("Encoder file not found. Proceeding without encoder.")
    #
    # # Check and load the scaler if it exists
    # if os.path.exists('scaler.pkl'):
    #     scaler = joblib.load('scaler.pkl')
    #     print(f"Scaler loaded: {type(scaler).__name__}")
    # else:
    #     print("Scaler file not found. Proceeding without scaler.")
    #
    # #validate_datasets(real_data, synthetic_data)
    #
    # # Convert DataFrames to NumPy arrays
    # real_data_np = real_data.to_numpy()
    # synthetic_data_np = synthetic_data.to_numpy()
    #
    # # Calculate DCR using Euclidean distance
    # dcr_euclidean = calculate_dcr(real_data_np, synthetic_data_np, metric='euclidean')
    # print(f"Average DCR (Euclidean): {dcr_euclidean}")
    #
    # # Calculate DCR using Manhattan (Cityblock) distance
    # dcr_manhattan = calculate_dcr(real_data_np, synthetic_data_np, metric='cityblock')
    # print(f"Average DCR (Manhattan): {dcr_manhattan}")
    #
    # # Calculate DCR using Hamming distance (ensure BINARY data for this metric)
    # if encoder is not None:
    #     categorical_cols = encoder.inverse_transform(synthetic_data_np[:, :len(encoder.get_feature_names_out())])
    #     synthetic_data_cat = np.concatenate([categorical_cols, synthetic_data_np[:, len(encoder.get_feature_names_out()):]], axis=1)
    # else:
    #     synthetic_data_cat = synthetic_data_np
    #
    # dcr_hamming = calculate_dcr(real_data_np, synthetic_data_cat, metric='hamming')
    # print(f"Average DCR (Hamming): {dcr_hamming}")

if __name__ == "__main__":
    main()