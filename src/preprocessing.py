import pandas as pd
import os
from utils import (
    PATH_DATA,
    TARGET,
    NUMERICAL_COLs,
    CATEGORICAL_COLs,
    ENCODER_CAT,
    ENCODER_NUM,
)
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


def get_data_X_y(path=PATH_DATA, target=TARGET):
    """
    retrieve the data from the path in the arg

    Args:
        path (str, optional): path to the csv. Defaults to PATH_DATA.
        target (list, optional): name of target col. Defaults to TARGET.

    Returns:
        df, array: X and y for the model
    """ ""
    # GET DATA
    data_df = pd.read_csv(os.path.join(path, "data\\data_income.csv"))
    data_df = pd.read_csv(os.path.join(path, "data\\data_income.csv"))
    data_df = data_df.dropna()

    # GET TARGET VS Y
    X = data_df.drop(target, axis=1)
    y = data_df[target]
    y = y.replace({">50K": 1, "<=50K": 0})

    return X, y


def split_train_test(X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """_summary_
    splitting the data 80%-20% for training and testing
    Args:
        X (pd.DataFrame): Data for training
        y (pd.DataFrame): target

    Returns:
        pd.DataFrame: dict of 4 dataframes X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


categorical_preprocessor = ENCODER_CAT
numeric_preprocessor = ENCODER_NUM
categorical_cols = CATEGORICAL_COLs
numerical_cols = NUMERICAL_COLs


def preprocessor(
    encoder_cat=categorical_preprocessor,
    encoder_num=numeric_preprocessor,
    cat_cols=categorical_cols,
    num_cols=numerical_cols,
):
    """
    Create a column transformer to scale the numerical features and encode the catorical features


    Args:
        encoder_cat (_type_, optional): OneHot or LabelEncoder. Defaults to categorical_preprocessor.
        encoder_num (_type_, optional): Scaler or "passthrough". Defaults to numeric_preprocessor.
        cat_cols (list, optional): list of categorical columns of X. Defaults to CATEGORICAL_COLs from utils.py
        num_cols (list, optional): list of categorical columns of X. Defaults to NUMERICAL_COLs from utils.py.
     Returns:
        obj: Return a preprocessor to insert in the pipeline
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", encoder_cat, cat_cols),
            ("num", encoder_num, num_cols),
        ],
        remainder="passthrough",
    )
    return preprocessor
