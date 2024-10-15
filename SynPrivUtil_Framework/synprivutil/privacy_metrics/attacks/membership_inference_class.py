import pandas as pd
from sdv.metadata import SingleTableMetadata
from tapas.datasets import Dataset, TabularDataset, DataDescription
from tapas.generators import Generator
from tapas.threat_models import BlackBoxKnowledge, TargetedMIA, NoBoxKnowledge, AuxiliaryDataKnowledge

from SynPrivUtil_Framework.synprivutil.models.models import CTGANModel
from SynPrivUtil_Framework.synprivutil.privacy_metrics import PrivacyMetricCalculator
from tapas.attacks import ClosestDistanceMIA, ProbabilityEstimationAttack, GroundhogAttack, ShadowModellingAttack, \
    LocalNeighbourhoodAttack, LpDistance
from tapas.threat_models import mia

class CTGAN_TAPAS(Generator):
    def __init__(self, filepath):
        super().__init__()
        self.ctgan_model = CTGANModel.load_model(filepath)
    def fit(self, dataset, **kwargs):
        self.ctgan_model.fit(dataset)
    def generate(self, num_samples, random_state=None):
        return self.ctgan_model.sample(num_samples)

    @property
    def label(self):
        return "CTGAN_TAPAS"

class MembershipInferenceCalculator(PrivacyMetricCalculator):
    def __init__(self):
        pass
    def evaluate(self):
        original_data = pd.read_csv("/Users/ksi/Development/Bachelorthesis/insurance.csv")
        synthetic_data = pd.read_csv("/Users/ksi/Development/Bachelorthesis/SynPrivUtil_Framework/synprivutil/models/insurance_datasets/random_sample.csv")

        data_knowledge = AuxiliaryDataKnowledge(
            dataset=TabularDataset(original_data, DataDescription()),
            auxiliary_split=0.5,  # Customize based on your attack setup
            num_training_records=1000,  # Adjust according to the data
        )
        # We use the same setup as groundhog_census.py.
        sdg_knowledge = BlackBoxKnowledge(
            generator=CTGAN_TAPAS("/Users/ksi/Development/Bachelorthesis/SynPrivUtil_Framework/synprivutil/models/diabetes_datasets/ctgan_model.pkl"),  # No generator is needed, as we already have synthetic data
            num_synthetic_records=1000
        )
        threat_model = TargetedMIA(
            attacker_knowledge_data=data_knowledge,
            target_record=targets,  # except with two targets!
            attacker_knowledge_generator=NoBoxKnowledge(),
            replace_target=True,
        )


    closest_distance_mia = ClosestDistanceMIA(distance=LpDistance())
    re = closest_distance_mia.attack(datasets=[synthetic_data])

    # Convert DataFrame to TAPAS Dataset
    def convert_to_tapas_dataset(df, name):
        return TabularDataset(data=df, description=DataDescription())

    # Convert the original and synthetic data to TAPAS Datasets
    original_dataset = convert_to_tapas_dataset(original_data, "original_data")
    synthetic_dataset = convert_to_tapas_dataset(synthetic_data, "synthetic_data")

