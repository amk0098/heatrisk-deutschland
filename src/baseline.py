import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    average_precision_score
)



def always_negative_baseline(y: pd.Series) -> np.ndarray:
    return np.zeros(len(y), dtype=int)


def persistence_baseline(txk: pd.Series) -> np.ndarray:
    return (txk >= 30.0).astype(int).to_numpy()


def seasonal_baseline(dates: pd.Series) -> np.ndarray:
    summer = dates.dt.month.isin([6, 7, 8])

    return summer.astype(int).to_numpy()


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

    _, y_test = split_features_target(test)

    predictions = seasonal_baseline(test["MESS_DATUM"])

    print("Number of test observations:", len(y_test))
    print("Number of predictions:", len(predictions))

    print("\nActual test target:")
    print(y_test.value_counts())

    print("\nPredicted target:")
    print(pd.Series(predictions).value_counts())

    print("\nSeasonal baseline metrics:")

    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Precision:", precision_score(y_test, predictions, zero_division=0))
    print("Recall:", recall_score(y_test, predictions, zero_division=0))
    print("F1:", f1_score(y_test, predictions, zero_division=0))
    print(
        "Balanced accuracy:",
        balanced_accuracy_score(y_test, predictions),
    )