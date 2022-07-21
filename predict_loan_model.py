from dataclasses import asdict

import pandas as pd
import joblib

from process_loan_data import process_loan_raw_data

df = pd.read_csv("data/prediction_rows/loan_data.csv")

model = joblib.load("models/loan_model.joblib")

raw_input = df.iloc[0]

predict_input = process_loan_raw_data(raw_input.to_dict())

prediction = model.predict(pd.DataFrame([asdict(predict_input)]))
print(prediction)
