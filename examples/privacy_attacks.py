import pandas as pd

from privacy_utility_framework.privacy_utility_framework.privacy_metrics import InferenceCalculator


def inference_example():
    synthetic_datasets = ["copulagan", "ctgan", "gaussian_copula", "gmm", "tvae", "random"]
    original_datasets =["insurance"]

    for orig in original_datasets:
        for syn in synthetic_datasets:
            original_data = pd.read_csv(f"../examples/{orig}_datasets/train/{orig}.csv")
            synthetic_data = pd.read_csv(f"../examples/{orig}_datasets/syn_on_train/{syn}_sample.csv")
            control = pd.read_csv(f"../examples/{orig}_datasets/test/{orig}.csv")
            columns = original_data.columns
            results = []

            # Iterate over all columns as secret, the rest as aux_cols and compute the individual risks
            # In the thesis report, the risk with the highest value was chosen and added to the results table
            for secret in columns:
                aux_cols = [col for col in columns if col != secret]
                test_sing = InferenceCalculator(original_data, synthetic_data, aux_cols=aux_cols, secret=secret,
                                                control=control)
                t = test_sing.evaluate()
                results.append((secret, t))
            print(f"~~~Synthetic Dataset generated with: {syn}~~~")
            print(results)


inference_example()