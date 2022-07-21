import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as LR
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

loans = pd.read_csv("data/loan_default_train_data.csv")

prediction_rows = loans.sample(frac=0.1, random_state=42)
prediction_rows.to_csv("data/prediction_rows/loan_data.csv", index=False)
df = loans[~loans.index.isin(prediction_rows.index)].reset_index(drop=True)

y = df["Credit Default"]

columns = [
    "Annual Income",
    "Number of Open Accounts",
    "Term",
    "Credit Score",
    "Home Ownership",
    "Current Loan Amount",
    "Current Credit Balance",
]

df_x = df[columns].copy()

df_x["ratio_loan_over_credit"] = (
    df_x["Current Loan Amount"] / df_x["Current Credit Balance"]
)


feature_columns = [
    "Annual Income",
    "Number of Open Accounts",
    "Credit Score",
    "ratio_loan_over_credit",
]

X = pd.concat(
    [
        pd.get_dummies(df_x["Term"], prefix="term", prefix_sep="_"),
        pd.get_dummies(
            df_x["Home Ownership"],
            prefix="home_ownership",
            prefix_sep="_",
        ),
        df_x[feature_columns],
    ],
    axis=1,
)

column_name_mapping = dict(
    zip(
        [
            "term_Long Term",
            "term_Short Term",
            "home_ownership_Have Mortgage",
            "home_ownership_Home Mortgage",
            "home_ownership_Own Home",
            "home_ownership_Rent",
            "Annual Income",
            "Number of Open Accounts",
            "Credit Score",
            "ratio_loan_over_credit",
        ],
        [
            "term_long_term",
            "term_short_term",
            "home_ownership_have_mortage",
            "home_ownership_home_mortage",
            "home_ownership_own_home",
            "home_ownership_rent",
            "anual_income",
            "number_of_open_accounts",
            "credit_score",
            "ratio_loan_over_credit",
        ],
    )
)

X.columns = X.columns.map(column_name_mapping)

X = X.drop("term_long_term", axis=1)
X = X.drop("home_ownership_rent", axis=1)

X.loc[X.isna().any(axis=1), "anual_income"] = X.anual_income.dropna().mean().round()
X.loc[X.isna().any(axis=1), "credit_score"] = X.credit_score.dropna().mean().round()
X.loc[X.isin([np.inf]).any(axis=1), "ratio_loan_over_credit"] = X[
    ~X.isin([np.inf]).any(1)
].ratio_loan_over_credit.mean()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = LR()
model.fit(X_train, y_train)

f1, accuracy = f1_score(model.predict(X_test), y_test), accuracy_score(
    model.predict(X_test), y_test
)
print("f1", f1, "accuracy", accuracy)

joblib.dump(model, "models/loan_model.joblib")
