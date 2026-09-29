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

if __name__ == "__main__":
    data_dir = Path("data/raw")

    df = load_all_stations(data_dir)

    print(df.head())
    print("\nRows:", len(df))
    print("\nColumns:", df.columns.tolist())
    print("\nCities:")
    print(df["city"].value_counts())
    print("\nFirst 10 rows:")
    print(df.head(10))
    print("\nLast 10 rows:")
    print(df.tail(10))  