import pandas as pd

from privacy_utility_framework.privacy_utility_framework.privacy_metrics import DCRCalculator, NNDRCalculator
from privacy_utility_framework.privacy_utility_framework.privacy_metrics.distance.adversarial_accuracy_class import \
    AdversarialAccuracyCalculator, AdversarialAccuracyCalculator_NN


def dcr_example():
    print("~~~~~~~~~DCR EXAMPLE~~~~~~~~~~")
    original_data = pd.read_csv("../datasets/original/diabetes.csv")
    synthetic_data = pd.read_csv(
        "../datasets/synthetic/diabetes_datasets/ctgan_sample.csv")

    test_dcr_calculator = DCRCalculator(original_data,
                                        synthetic_data, weights=[1,1,1,1,1,1,1,1,1])
    test_dcr = test_dcr_calculator.evaluate()
    print(f'DCR (diabetes, ctgan): {test_dcr}')


def nndr_example():
    print("~~~~~~~~~NNDR EXAMPLE~~~~~~~~~~")
    synthetic_datasets = ["copulagan", "ctgan", "gaussian_copula", "gmm", "tvae", "random"]
    original_datasets =["diabetes",  "insurance"]

    for orig in original_datasets:
        for syn in synthetic_datasets:
            original_data = pd.read_csv(f"../datasets/original/{orig}.csv")
            synthetic_data = pd.read_csv(
                f"../datasets/synthetic/{orig}_datasets/{syn}_sample.csv")
            test_nndr_calculator = NNDRCalculator(original_data, synthetic_data)
            test_nndr = test_nndr_calculator.evaluate()
            print(f'NNDR {orig, syn}: {test_nndr}')

def nnaa_example():
    print("~~~~~~~~~NNAA EXAMPLE~~~~~~~~~~")
    synthetic_datasets = ["copulagan", "ctgan", "gaussian_copula", "gmm", "tvae", "random"]
    original_datasets =["diabetes", "insurance"]

    for orig in original_datasets:
        for syn in synthetic_datasets:
            original_data = pd.read_csv(f"../datasets/original/{orig}.csv")
            synthetic_data = pd.read_csv(
                f"../datasets/synthetic/{orig}_datasets/{syn}_sample.csv")
            print(f'~~~~~~Adversarial Accuracy CDIST~~~~~~ {orig, syn}')

            calculator_cdist = AdversarialAccuracyCalculator(original_data, synthetic_data)
            nnaa1 = calculator_cdist.evaluate()
            print(nnaa1)

            print(f'~~~~~~Adversarial Accuracy NN~~~~~~ {orig, syn}')
            calculator_nn = AdversarialAccuracyCalculator_NN(original_data, synthetic_data)
            nnaa2 = calculator_nn.evaluate()
            print(nnaa2)

dcr_example()
nndr_example()
nnaa_example()