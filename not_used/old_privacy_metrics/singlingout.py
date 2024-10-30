from anonymeter.evaluators import SinglingOutEvaluator
import pandas as pd


def evaluate_singling_out_risk(real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> SinglingOutEvaluator:
    """
    Evaluate the singling out risk between real and synthetic datasets.

    Parameters:
        real_data (pd.DataFrame): The original dataset.
        synthetic_data (pd.DataFrame): The synthetic dataset.

    Returns:
        dict: A dictionary with singling out risk evaluation metrics.
    """
    evaluator = SinglingOutEvaluator(real_data, synthetic_data)
    singling_out_risk = evaluator.evaluate()
    return singling_out_risk


if __name__ == "__main__":
    real_data = pd.read_csv('/diabetes_transformed.csv')
    synthetic_data = pd.read_csv('/synthetic_data_transformed.csv')
    singling_out_risk_evaluator = evaluate_singling_out_risk(real_data, synthetic_data)
    risk = singling_out_risk_evaluator.risk()
    print(f"Singling Out Risk Evaluation:\n{risk}")
