import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

import xgboost as xgb
from xgboost import XGBClassifier


def train_model_xgb(
    X: pd.DataFrame, y: pd.DataFrame, preprocessor: obj
) -> XGBClassifier:
    """_summary_
    trainning the XGBoost Classifier model
    Args:
        X (pd.DataFrame): input data not the train/ test
        y (pd.DataFrame): target array
        preprocessor (obj): scaler and category encoder for preprocessing

    Returns:
        XGBClassifier: fitted model
    """

    xgb_model = XGBClassifier(random_state=42)

    xgb_pipeline = Pipeline([("preprocessor", preprocessor), ("model", xgb_model)])
    # Fit the pipeline (including preprocessing) to the training data
    xgb_pipeline.fit(X, y)
    return xgb_pipeline


def predict(X: pd.DataFrame, model: XGBClassifier) -> np.ndarray:
    """_summary_
    Predicting the target for a X_test data input

    Args:
        X (pd.DataFrame): X_test to predict
        model (XGBClassifier): fitted model

    Returns:
        np.ndarray: y_pred modelled by the xgb classifier
    """
    return model.predict(X)


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """_summary_
    Evaluate the prediction of the model

    Args:
        y_true (np.ndarray): true values of the target
        y_pred (np.ndarray): predicted values of the target

    Returns:
        float, float: two scores to evaluate the prediction, f1 and accuracy
    """
    return f1_score(y_true, y_pred), accuracy_score(y_true, y_pred)
