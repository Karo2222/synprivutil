from anonymeter.evaluators import InferenceEvaluator
import pandas as pd


def evaluate_inference_risk(real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> InferenceEvaluator:
    """
    Evaluate the inference risk between real and synthetic datasets.

    Parameters:
        real_data (pd.DataFrame): The original dataset.
        synthetic_data (pd.DataFrame): The synthetic dataset.

    Returns:
        dict: A dictionary with inference risk evaluation metrics.
    """
    evaluator = InferenceEvaluator(real_data, synthetic_data, ['Age', 'Glucose', 'BloodPressure'], 'BMI')
    inference_risk = evaluator.evaluate()
    return inference_risk


if __name__ == "__main__":
    real_data = pd.read_csv('/diabetes_transformed.csv')
    synthetic_data = pd.read_csv('/synthetic_data_transformed.csv')
    inference_risk_evaluator = evaluate_inference_risk(real_data, synthetic_data)
    risk = inference_risk_evaluator.risk()
    print(f"Inference Risk Evaluation:\n{risk} (0: maximum privacy, 1: records are being leaked)")
