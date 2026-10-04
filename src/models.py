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





if __name__ == "__main__":
    from pathlib import Path

    from src.data_loader import (
        load_all_stations,
        add_target,
        remove_missing_targets,
    )
    from src.features import (
        add_lag_features,
        remove_missing_features,
    )
    from src.split import (
        chronological_split,
        split_features_target,
    )

    data_dir = Path("data/raw")

    df = load_all_stations(data_dir)
    df = add_target(df)
    df = remove_missing_targets(df)
    df = add_lag_features(df)
    df = remove_missing_features(df)

    train, validation, test = chronological_split(df)

    X_train, y_train = split_features_target(train)
    X_validation, y_validation = split_features_target(validation)
    X_test, y_test = split_features_target(test)

    model = train_random_forest(X_train, y_train)

validation_probabilities = model.predict_proba(
    X_validation
)[:, 1]

thresholds = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
]

threshold_results = evaluate_thresholds(
    y_validation,
    validation_probabilities,
    thresholds,
)

print("\nRandom Forest validation threshold results:")
print(threshold_results)

selected_threshold = (
    threshold_results
    .sort_values("f1", ascending=False)
    .iloc[0]["threshold"]
)

print("\nSelected threshold:", selected_threshold)

test_probabilities = model.predict_proba(
    X_test
)[:, 1]

test_predictions = predict_with_threshold(
    test_probabilities,
    selected_threshold,
)

print("\nFinal Random Forest test metrics:")

print(
    "Accuracy:",
    accuracy_score(y_test, test_predictions),
)

print(
    "Precision:",
    precision_score(
        y_test,
        test_predictions,
        zero_division=0,
    ),
)

print(
    "Recall:",
    recall_score(
        y_test,
        test_predictions,
        zero_division=0,
    ),
)

print(
    "F1:",
    f1_score(
        y_test,
        test_predictions,
        zero_division=0,
    ),
)

print(
    "Balanced accuracy:",
    balanced_accuracy_score(
        y_test,
        test_predictions,
    ),
)

print(
    "Average precision:",
    average_precision_score(
        y_test,
        test_probabilities,
    ),
)

print("\nConfusion matrix:")
print(confusion_matrix(y_test, test_predictions))