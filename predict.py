from dataclasses import asdict

from process_data.process_loan_data import process_loan_raw_data
from process_data.process_titanic_data import process_titanic_raw_data
import pandas as pd

import joblib

model_registry = joblib.load("configs/model_registry/model_registry.joblib")

products = ["titanic", "loans"]
for product in products:
    model_meta_data = [i for i in model_registry if i.product_version == product][0]
    model = joblib.load(model_meta_data.model_path[3:])

    prediction_rows = None
    if product == "loans":
        prediction_rows = pd.read_csv("data/prediction_rows/loan_data.csv")
    elif product == "titanic":
        prediction_rows = pd.read_csv("data/prediction_rows/titanic_data.csv")

    raw_input = prediction_rows.iloc[0]
    print("product", product)
    print("raw input")
    print(raw_input.to_dict())

    predict_input = None
    if product == "loans":
        predict_input = process_loan_raw_data(raw_input.to_dict())
    elif product == "titanic":
        predict_input = process_titanic_raw_data(raw_input.to_dict())

    print("model input")
    print(predict_input)

    prediction = model.predict(pd.DataFrame([asdict(predict_input)]))
    print(prediction)
