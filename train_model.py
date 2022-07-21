from model_registry import ModelRegistryEntry

from train_titanic import train_titanic_model

MODEL_REGISTRY = []

data_path = "data/titanic_survive_train.csv"
model_path = "models/titanic_model.joblib"
model, f1, accuracy = train_titanic_model(data_path, model_path)
titanic_model_registry_entry = ModelRegistryEntry(
    product_version="titanic",
    data_version="??",
    model_metrics=dict(f1=f1, accuracy=accuracy),
    model_path=model_path,
    model_version="0.0.1",
)

MODEL_REGISTRY.append(titanic_model_registry_entry)


print(MODEL_REGISTRY)
