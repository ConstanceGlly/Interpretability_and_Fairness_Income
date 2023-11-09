import pandas as pd
import numpy as np

from modelling import train_model_xgb, predict, evaluate_model
from preprocessing import get_data_X_y, split_train_test, preprocessor
from utils import (
    PATH_DATA,
    TARGET,
    NUMERICAL_COLs,
    CATEGORICAL_COLs,
    ENCODER_CAT,
    ENCODER_NUM,
)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler

categorical_preprocessor = ENCODER_CAT
numeric_preprocessor = ENCODER_NUM
categorical_cols = CATEGORICAL_COLs
numerical_cols = NUMERICAL_COLs


def predict_eval() -> dict:
    """_summary_
    Full workflow of the prediction of the revenue, all inputs are optionnal and need not to be changed

    Returns:
        dict: y_pred the target predicted by the model and the f1 score of the prediction
    """

    # Get the data from the path
    X, y = get_data_X_y(PATH_DATA, TARGET)

    # Split the data
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # Prepare preprocessor
    preprocessor_xgb = preprocessor(
        categorical_preprocessor, numeric_preprocessor, categorical_cols, numerical_cols
    )

    # Train the model
    model_xgb = train_model_xgb(X_train, y_train, preprocessor_xgb)

    # Make the prediction
    y_pred = predict(y_train, model_xgb)

    # Evaluate the model
    score_f1, score_acc = evaluate_model(y_pred, y_test)

    return y_pred, score_f1
