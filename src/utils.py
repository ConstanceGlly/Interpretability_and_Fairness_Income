from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler

CATEGORICAL_COLs =[
    "occupation",
    "workclass",
    "marital-status",
    "relationship",
    "race",
    "native-country",
    "gender",
    "income",
]
NUMERICAL_COLs = [
    "age",
    "fnlwgt",
    "educational-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

ENCODER_CAT = OneHotEncoder(drop="if_binary", handle_unknown="ignore")
ENCODER_NUM = StandardScaler()

TARGET = ["income"]

DROP_COL = []

PATH_DATA = "C:\Users\const\OneDrive\Documents\GitHub\InterpretabilityAndFairness_Tinder\data"