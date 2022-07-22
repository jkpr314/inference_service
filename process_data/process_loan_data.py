from dataclasses import dataclass

import numpy as np
import pandas as pd

# prepare for prediction


@dataclass
class LoanInput:
    term_short_term: int
    home_ownership_have_mortage: int
    home_ownership_home_mortage: int
    home_ownership_own_home: int
    annual_income: float
    number_of_open_accounts: int
    credit_score: float
    ratio_loan_over_credit: float


def process_term(value):
    if value == "Short Term":
        return 1
    return 0


def process_home_ownership(value):
    if value == "Home Mortgage":
        return 0, 1, 0
    elif value == "Have Mortgage":
        return 1, 0, 0
    elif value == "Own Home":
        return 0, 0, 1
    return 0, 0, 0


def process_income(value):
    if value != value:
        return 1364823.0
    return value


def process_number_of_open_accounts(value):
    return value


def process_credit_score(value):
    if value != value:
        return 1137.0
    return value


def process_ratio_loan_over_credit(data):
    ratio = data.get("Current Loan Amount") / data.get("Current Credit Balance")
    if np.isinf(ratio):
        return 500.288
    return ratio


# prepare for training


def process_loan_raw_data(data: dict) -> LoanInput:
    term = process_term(data.get("Term"))
    (
        home_ownership_have_mortage,
        home_ownership_home_mortage,
        home_ownership_own_home,
    ) = process_home_ownership(data.get("Home Ownership"))
    annual_income = process_income(data.get("Annual Income"))
    number_of_open_accounts = process_number_of_open_accounts(
        data.get("Number of Open Accounts")
    )
    credit_score = process_credit_score(data.get("Credit Score"))
    ratio_loan_over_credit = process_ratio_loan_over_credit(data)
    return LoanInput(
        term_short_term=term,
        home_ownership_have_mortage=home_ownership_have_mortage,
        home_ownership_home_mortage=home_ownership_home_mortage,
        home_ownership_own_home=home_ownership_own_home,
        annual_income=annual_income,
        number_of_open_accounts=number_of_open_accounts,
        credit_score=credit_score,
        ratio_loan_over_credit=ratio_loan_over_credit,
    )


def prepare_loan_data(df):
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
                "annual_income",
                "number_of_open_accounts",
                "credit_score",
                "ratio_loan_over_credit",
            ],
        )
    )

    X.columns = X.columns.map(column_name_mapping)

    X = X.drop("term_long_term", axis=1)
    X = X.drop("home_ownership_rent", axis=1)

    X.loc[X.isna().any(axis=1), "annual_income"] = (
        X.annual_income.dropna().mean().round()
    )
    X.loc[X.isna().any(axis=1), "credit_score"] = X.credit_score.dropna().mean().round()
    X.loc[X.isin([np.inf]).any(axis=1), "ratio_loan_over_credit"] = X[
        ~X.isin([np.inf]).any(1)
    ].ratio_loan_over_credit.mean()
    return X, y
