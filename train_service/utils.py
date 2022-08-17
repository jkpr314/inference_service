import joblib
import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

from process_data.process_loan_data import prepare_loan_data
from process_data.process_titanic_data import prepare_titanic_data


def load_data(data_path, product):
    loans = pd.read_csv(data_path)

    prediction_rows = loans.sample(frac=0.1, random_state=42)
    if product == "loans":
        prediction_rows.to_csv("../data/prediction_rows/loan_data.csv", index=False)
    elif product == "titanic":
        prediction_rows.to_csv("../data/prediction_rows/titanic_data.csv", index=False)
    df = loans[~loans.index.isin(prediction_rows.index)].reset_index(drop=True)
    return df


def train_base_model(X_train, y_train):
    model = LR()
    model.fit(X_train, y_train)
    return model


def get_model_scores(model, X_test, y_test):
    f1, accuracy = f1_score(model.predict(X_test), y_test), accuracy_score(
        model.predict(X_test), y_test
    )
    return f1, accuracy


def save_model(model, model_path):
    joblib.dump(model, model_path)


def train_model(data_path, model_path, product):
    df = load_data(data_path, product)
    if product == "loans":
        prepare_data = prepare_loan_data
    elif product == "titanic":
        prepare_data = prepare_titanic_data
    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    model = train_base_model(X_train, y_train)
    f1, accuracy = get_model_scores(model, X_test, y_test)
    save_model(model, model_path)

    return f1, accuracy
