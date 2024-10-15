from scipy.stats import spearmanr

from SynPrivUtil_Framework.synprivutil.privacy_metrics.distance.adversarial_accuracy_class import AdversarialAccuracyCalculator
from SynPrivUtil_Framework.synprivutil.utility_metrics.statistical.wasserstein import WassersteinCalculator
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




    orig = pd.read_csv("/Users/ksi/Development/Bachelorthesis/transformed_SD2011_selected_columns.csv")
    syn = pd.read_csv("/Users/ksi/Development/Bachelorthesis/transformed_syn_SD2011_selected_columns.csv")


    orig = pd.read_csv("/Users/ksi/Development/Bachelorthesis/diabetes_transformed_ctgan.csv")
    syn = pd.read_csv("/Users/ksi/Development/Bachelorthesis/synthetic_data_transformed_ctgan.csv")

    # Calculate descriptive statistics for each numerical column
    desc_stats_original = orig.describe().loc[['min', 'max', '50%', 'mean', 'std']]
    desc_stats_synthetic = syn.describe().loc[['min', 'max', '50%', 'mean', 'std']]
    print(desc_stats_original)
    print(desc_stats_synthetic)

    # Flatten the statistics and compute Spearman's Rho correlation
    flattened_original = desc_stats_original.values.flatten()
    flattened_synthetic = desc_stats_synthetic.values.flatten()

    # Spearman's Rho calculation
    rho, p_value = spearmanr(flattened_original, flattened_synthetic)

    print(f"Spearman's Rho: {rho}")

    # Evaluate all added metrics
    results = p.evaluate_all()
    for key, value in results.items():
        print(f"{key}: {value}")

    t = WassersteinCalculator(orig, syn)
    print(f"~~~~~~~~~~~~~ Wasserstein might take a longer time. ~~~~~~~~~~~~~")
    print(f"Wasserstein: {t.evaluate()}")


if __name__ == "__main__":
    main()