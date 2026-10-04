import pandas as pd



def calculate_rolling_3(s: pd.Series) -> pd.Series:
    return s.shift(1).rolling(3).mean()

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["TXK_lag_1"] = (
        df.groupby("city")["TXK"]
        .shift(1)
    )

    df["TXK_rolling_3"] = (
        df.groupby("city")["TXK"]
        .transform(calculate_rolling_3)
    )

    return df


def remove_missing_features(df: pd.DataFrame) -> pd.DataFrame:
    required_history_features = [
        "TXK_lag_1",
        "TXK_rolling_3",
    ]

    df = df.dropna(
        subset=required_history_features
    ).copy()

    return df


def remove_missing_features(df: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [
        "TMK",
        "TXK",
        "TNK",
        "VPM",
        "UPM",
        "TXK_lag_1",
        "TXK_rolling_3",
    ]

    df = df.dropna(subset=feature_columns).copy()

    return df


if __name__ == "__main__":
    from pathlib import Path
    from src.data_loader import (
        load_all_stations,
        add_target,
        remove_missing_targets,
    )

    data_dir = Path("data/raw")

    df = load_all_stations(data_dir)
    df = add_target(df)
    df = remove_missing_targets(df)

    df = add_lag_features(df)
    df = remove_missing_features(df)
    features = remove_missing_features(df)
    print(
    df[
        ["MESS_DATUM","city","TXK","TXK_lag_1","TXK_rolling_3","heat_day_next",]].head(10))

    print("\nSelected features:")
    print(features.head())

    print("\nFeature columns:")
    print(features.columns.tolist())

    print("\nShape:")
    print(features.shape)

    print("\nAfter removing missing engineered features:")
    print(features.head())
    print("\nShape:")
    print(features.shape)
    print("\nDate range:")
    print(df["MESS_DATUM"].min())
    print(df["MESS_DATUM"].max())

    print("\nRows by period:")

    train = df[df["MESS_DATUM"] <= "2020-12-31"]
    validation = df[
    (df["MESS_DATUM"] >= "2021-01-01")
    & (df["MESS_DATUM"] <= "2023-12-31")
        ]
    test = df[df["MESS_DATUM"] >= "2024-01-01"]

    print("Train:", train.shape)
    print("Validation:", validation.shape)
    print("Test:", test.shape)


    print("\nHeat days by period:")
    print("Train:", train["heat_day_next"].sum())
    print("Validation:", validation["heat_day_next"].sum())
    print("Test:", test["heat_day_next"].sum())
    print("\nHeat-day rate by period:")
    print("Train:", train["heat_day_next"].mean())
    print("Validation:", validation["heat_day_next"].mean())
    print("Test:", test["heat_day_next"].mean())