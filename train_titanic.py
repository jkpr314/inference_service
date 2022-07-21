import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import f1_score, accuracy_score
import joblib


def load_data(data_path):
    titanic = pd.read_csv(data_path)

    prediction_rows = titanic.sample(frac=0.1, random_state=42)
    prediction_rows.to_csv("data/prediction_rows/titanic_data.csv", index=False)

    return titanic[~titanic.index.isin(prediction_rows.index)]


def prepare_data(df):

    df = df.drop("PassengerId", axis=1)

    y = df.Survived
    df = df.drop("Survived", axis=1)

    X = pd.concat(
        [
            pd.get_dummies(df["Pclass"], prefix="Pclass", prefix_sep="_"),
            pd.get_dummies(
                df["Sex"],
                prefix="Sex",
                prefix_sep="_",
            ),
            df[["Age", "Fare"]],
        ],
        axis=1,
    )

    X.loc[X.isna().any(axis=1), "Age"] = X.Age.dropna().mean().round()

    X = X.drop("Pclass_3", axis=1)
    X = X.drop("Sex_female", axis=1)

    X.columns = [i.lower() for i in X.columns]

    return X, y


# fix missing values


def train_model(X_train, y_train):

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


def train_titanic_model(data_path, model_path):

    df = load_data(data_path)

    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    model = train_model(X_train, y_train)
    f1, accuracy = get_model_scores(model, X_test, y_test)
    save_model(model, model_path)

    return f1, accuracy
