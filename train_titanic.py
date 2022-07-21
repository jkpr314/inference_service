import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import f1_score, accuracy_score
import joblib

titanic = pd.read_csv("data/titanic_survive_train.csv")

prediction_rows = titanic.sample(frac=0.1, random_state=42)
prediction_rows.to_csv("data/prediction_rows/titanic_data.csv", index=False)

df = titanic[~titanic.index.isin(prediction_rows.index)]

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

# fix missing values
X.loc[X.isna().any(axis=1), "Age"] = X.Age.dropna().mean().round()

X = X.drop("Pclass_3", axis=1)
X = X.drop("Sex_female", axis=1)

X.columns = [i.lower() for i in X.columns]

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = LR()
model.fit(X_train, y_train)

model.predict(X_test)


f1, accuracy = f1_score(model.predict(X_test), y_test), accuracy_score(
    model.predict(X_test), y_test
)
print("f1", f1, "accuracy", accuracy)

joblib.dump(model, "models/titanic_model.joblib")
