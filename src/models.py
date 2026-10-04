import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier






def create_logistic_regression() -> Pipeline:
    model = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="mean"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
            ),
        ),
    ])

    return model








def create_random_forest() -> Pipeline:
    model = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="mean"),
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ])

    return model




def train_logistic_regression(X_train: pd.DataFrame,y_train: pd.Series,) -> Pipeline:

    model = create_logistic_regression()

    model.fit(X_train, y_train)

    return model






def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:

    model = create_random_forest()

    model.fit(X_train, y_train)

    return model




#come back to it
def evaluate_thresholds(
    y_true: pd.Series,
    probabilities,
    thresholds: list[float],
) -> pd.DataFrame:

    results = []

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)

        results.append({
            "threshold": threshold,
            "precision": precision_score(
                y_true,
                predictions,
                zero_division=0,
            ),
            "recall": recall_score(
                y_true,
                predictions,
                zero_division=0,
            ),
            "f1": f1_score(
                y_true,
                predictions,
                zero_division=0,
            ),
            "balanced_accuracy": balanced_accuracy_score(
                y_true,
                predictions,
            ),
        })

    return pd.DataFrame(results)






def predict_with_threshold(
    probabilities,
    threshold: float,
):
    return (probabilities >= threshold).astype(int)

