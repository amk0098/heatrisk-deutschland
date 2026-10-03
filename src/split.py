import pandas as pd

FEATURE_COLUMNS = [
    "TMK",
    "TXK",
    "TNK",
    "VPM",
    "UPM",
    "TXK_lag_1",
    "TXK_rolling_3",
]


def split_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:

    X = df[FEATURE_COLUMNS].copy()
    y = df["heat_day_next"].copy()

    return X , y




def chronological_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    train = df[df["MESS_DATUM"] <= "2020-12-31"].copy()

    validation = df[
        (df["MESS_DATUM"] >= "2021-01-01")
        & (df["MESS_DATUM"] <= "2023-12-31")
    ].copy()

    test = df[df["MESS_DATUM"] >= "2024-01-01"].copy()

    return train, validation, test




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

    print("\nX_train:")
    print(X_train.shape)

    print("\nY_train:")
    print(y_train.shape)

    print("\nX_train columns:")
    print(X_train.columns.tolist())

    print("\nY_train values:")
    print(y_train.value_counts())

    print("Train:", train.shape)
    print("Validation:", validation.shape)
    print("Test:", test.shape)

    print("\nDate ranges:")
    print("Train:", train["MESS_DATUM"].min(), "to", train["MESS_DATUM"].max())
    print(
        "Validation:",
        validation["MESS_DATUM"].min(),
        "to",
        validation["MESS_DATUM"].max(),
    )
    print("Test:", test["MESS_DATUM"].min(), "to", test["MESS_DATUM"].max())
