from dataclasses import dataclass

@dataclass
class ModelRegistryEntry:
    product_version: str
    data_version: str
    model_metrics: dict
    model_path: str
    model_version: str
