from dataclasses import dataclass


@dataclass
class TitanicInput:
    pclass_1: int
    pclass_2: int
    sex_male: int
    age: float
    fare: float


def process_pclass(pclass_value):
    if pclass_value == 1:
        return 1, 0
    elif pclass_value == 2:
        return 0, 1
    return 0, 0


def process_sex(sex_value):
    if sex_value == "male":
        return 1
    return 0


def process_age(age_value):
    if age_value != age_value:
        return 30.0
    return age_value


def process_titanic_raw_data(data: dict) -> TitanicInput:
    pclass_1, pclass_2 = process_pclass(data.get("Pclass"))
    sex_male = process_sex(data.get("Sex"))
    age = process_age(data.get("Age"))
    return TitanicInput(
        pclass_1=pclass_1,
        pclass_2=pclass_2,
        sex_male=sex_male,
        age=age,
        fare=data.get("Fare"),
    )
