from pathlib import Path
import pandas as pd
CITY_MAP = {
    "produkt_klima_tag_Berlin.txt": "Berlin",
    "produkt_klima_tag_Frankfurt.txt": "Frankfurt",
    "produkt_klima_tag_Hamburg.txt": "Hamburg",
    "produkt_klima_tag_Muenchen.txt": "Muenchen",
    "produkt_klima_tag_koeln.txt": "Koeln",
}
def load_station_data(file_path:Path)->pd.DataFrame:
    df = pd.read_csv(
        file_path,
        sep=";",
        na_values=["-999", "-999.0"],)
    
    df.columns = df.columns.str.strip()

    df["MESS_DATUM"] = pd.to_datetime(
        df["MESS_DATUM"],
        format="%Y%m%d",)
    df = df[df["MESS_DATUM"] >= "1990-01-01"]

    df["city"] = CITY_MAP[file_path.name]

    keep_columns = [
    "MESS_DATUM",
    "TMK",
    "TXK",
    "TNK",
    "VPM",
    "UPM",
    "city",]
    df = df[keep_columns]
    return df



def load_all_stations(data_dir: Path) -> pd.DataFrame:
    files = sorted(data_dir.glob("*.txt"))

    dataframes = []

    for file_path in files:
        df = load_station_data(file_path)
        dataframes.append(df)

    df = pd.concat(dataframes, ignore_index=True)

    df = df.sort_values(["city", "MESS_DATUM"]).reset_index(drop=True)

    return df

def add_target(df: pd.DataFrame) -> pd.DataFrame:
    next_txk = df.groupby("city")["TXK"].shift(-1)

    df["heat_day_next"] = (next_txk >= 30.0).astype("Int64")

    df.loc[next_txk.isna(), "heat_day_next"] = pd.NA

    return df



def remove_missing_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["heat_day_next"]).copy()

    df["heat_day_next"] = df["heat_day_next"].astype(int)

    return df

if __name__ == "__main__":
    data_dir = Path("data/raw")

    df = load_all_stations(data_dir)
    df = add_target(df)
    df = remove_missing_targets(df)

    print(df.head())

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nShape:")
    print(df.shape)

    print("\nTarget counts:")
    print(df["heat_day_next"].value_counts())