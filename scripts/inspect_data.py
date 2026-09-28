from pathlib import Path
import pandas as pd
DATA_DIR=Path("data/raw")
files = sorted(DATA_DIR.glob("*.txt"))
print(f"Found {len(files)} files.\n")
city_map = {
    "produkt_klima_tag_Berlin.txt": "Berlin",
    "produkt_klima_tag_Frankfurt.txt": "Frankfurt",
    "produkt_klima_tag_Hamburg.txt": "Hamburg",
    "produkt_klima_tag_Muenchen.txt": "Muenchen",
    "produkt_klima_tag_koeln.txt": "Koeln",
}
for file in files: 
    print("=" * 60)
    print(f"FILE: {file.name}")
    print("=" * 60)
    df = pd.read_csv(
    file,
    sep=";",
    na_values=["-999", "-999.0"],
)
    df.columns = df.columns.str.strip()
    df["city"] = city_map[file.name]
    print("\nColumn overview:")
    print(
    pd.DataFrame({
        "missing": df.isna().sum(),
        "missing_%": (df.isna().mean() * 100).round(2),
    })
    )
    df["MESS_DATUM"] = pd.to_datetime(df["MESS_DATUM"], format="%Y%m%d")
    df = df[df["MESS_DATUM"] >= "1990-01-01"]
    print("\nMissing temperature dates:")
    print(df.loc[df["TXK"].isna(), ["MESS_DATUM", "TXK"]].head(10))
    keep_columns = ["MESS_DATUM", "TMK", "TXK", "TNK", "VPM", "UPM", "city"]
    print("\nCandidate columns:")
    print(df[keep_columns].head())
    print("\nCandidate missing values:")
    print(df[keep_columns].isna().sum())
    """print("\nTemperature summary:")
    print(df[["TMK", "TXK", "TNK"]].describe())
    print("\nRaw column names:")
    print(df.columns.tolist())
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nStation IDs:")
    print(df["STATIONS_ID"].unique())

    print("\nDate range:")
    print(df["MESS_DATUM"].min(), "->", df["MESS_DATUM"].max())

    print("\nMissing values:")
    print(df.isna().sum())
    expected_days = (df["MESS_DATUM"].max() - df["MESS_DATUM"].min()).days + 1

    print("\nExpected vs actual days:")
    print(f"Expected calendar days: {expected_days:,}")
    print(f"Actual rows: {len(df):,}")
    print(f"Missing days: {expected_days - len(df):,}")

    print("\nDuplicate dates:")
    print(df["MESS_DATUM"].duplicated().sum())

    print("\nData types:")
    print(df.dtypes)"""
