from anonymeter.evaluators import LinkabilityEvaluator
import pandas as pd


def evaluate_linkability_risk(real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> LinkabilityEvaluator:
    """
    Evaluate the linkability risk between real and synthetic datasets.
    The LinkabilityEvaluator allows one to know how much the synthetic data will help an adversary who tries to link two other datasets based on a subset of attributes.

    For example, suppose that the adversary finds dataset A containing, among other fields,
    information about the profession and education of people, and dataset B containing some
    demographic and health related information. Can the attacker use the synthetic dataset
    to link these two datasets?

    Parameters:
        real_data (pd.DataFrame): The original dataset.
        synthetic_data (pd.DataFrame): The synthetic dataset.

    Returns:
        dict: A dictionary with linkability risk evaluation metrics.
    """
    evaluator = LinkabilityEvaluator(real_data, synthetic_data,
                                     (['Age', 'Glucose', 'BloodPressure'], ['Insulin', 'BMI', 'Age']))
    linkability_risk = evaluator.evaluate()
    return linkability_risk


if __name__ == "__main__":
    real_data = pd.read_csv('/diabetes_transformed.csv')
    synthetic_data = pd.read_csv('/synthetic_data_transformed.csv')
    linkability_risk_evaluator = evaluate_linkability_risk(real_data, synthetic_data)
    risk = linkability_risk_evaluator.risk()
    print(f"Linkability Risk Evaluation:\n{risk} (0: maximum privacy, 1: records are being leaked)")
