from model_registry import ModelRegistryEntry

from train_service.utils import train_model

MODEL_REGISTRY = []

data_path = "../data/titanic_survive_train.csv"
model_path = "../models/titanic_model.joblib"
f1, accuracy = train_model(data_path, model_path, product="titanic")

titanic_model_registry_entry = ModelRegistryEntry(
    product_version="titanic",
    data_version="??",
    model_metrics=dict(f1=f1, accuracy=accuracy),
    model_path=model_path,
    model_version="0.0.1",
)

MODEL_REGISTRY.append(titanic_model_registry_entry)

model_path = "../models/loan_model.joblib"
data_path = "../data/loan_default_train_data.csv"
f1, accuracy = train_model(data_path, model_path, product="loans")
loans_model_registry_entry = ModelRegistryEntry(
    product_version="loans",
    data_version="??",
    model_metrics=dict(f1=f1, accuracy=accuracy),
    model_path=model_path,
    model_version="0.0.1",
)

MODEL_REGISTRY.append(loans_model_registry_entry)


print(MODEL_REGISTRY)
