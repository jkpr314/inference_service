from dataclasses import dataclass

from train_service.train_model import train_all


@dataclass
class ModelRegistryEntry:
    product_version: str
    data_version: str
    f1_score: float
    accuracy: float
    model_path: str
    model_version: str


model_registry = train_all()
