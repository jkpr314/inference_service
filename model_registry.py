from dataclasses import dataclass


@dataclass
class ModelRegistryEntry:
    product_version: str
    data_version: str
    f1_score: float
    accuracy: float
    model_path: str
    model_version: str
