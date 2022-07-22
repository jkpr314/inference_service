from model_registry import ModelRegistryEntry

from train_service.utils import train_model

import joblib

def train_all():
    model_registry = []

    data_path = "../data/titanic_survive_train.csv"
    model_path = "../models/titanic_model.joblib"
    f1, accuracy = train_model(data_path, model_path, product="titanic")

    titanic_model_registry_entry = ModelRegistryEntry(
        product_version="titanic",
        data_version="??",
        f1_score=f1,
        accuracy=accuracy,
        model_path=model_path,
        model_version="0.0.1",
    )

    model_path = "../models/loan_model.joblib"
    data_path = "../data/loan_default_train_data.csv"
    f1, accuracy = train_model(data_path, model_path, product="loans")
    loans_model_registry_entry = ModelRegistryEntry(
        product_version="loans",
        data_version="??",
        f1_score=f1,
        accuracy=accuracy,
        model_path=model_path,
        model_version="0.0.1",
    )

    model_registry += [titanic_model_registry_entry, loans_model_registry_entry]

    return model_registry


if __name__ == "__main__":
    models = train_all()

    joblib.dump(models, '../configs/model_registry/model_registry.joblib')