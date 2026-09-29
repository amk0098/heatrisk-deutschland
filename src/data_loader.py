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

if __name__ == "__main__":
    data_dir = Path("data/raw")

    files = sorted(data_dir.glob("*.txt"))

    print(f"Found {len(files)} files.\n")

    for file_path in files:
        df = load_station_data(file_path)

        print(f"{df['city'].iloc[0]}:")
        print(f"  Rows: {len(df)}")
        print(f"  Date range: {df['MESS_DATUM'].min()} -> {df['MESS_DATUM'].max()}")
        print(f"  Columns: {df.columns.tolist()}")
        print()

