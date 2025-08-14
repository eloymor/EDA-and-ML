import os
import sqlite3
import json
import argparse
import pandas as pd
from sklearn.cluster import DBSCAN

DB_NAME = os.path.join("..", "etl_data.db")

with open(os.path.join("..", "data", "proximity_mapping.json"), "r") as f:
    ocean_proximity_map: dict[str, int] = json.load(f)

def extract_data(file_name: str) -> pd.DataFrame:
    path_from_root = os.path.join("..", "data", file_name)
    if not os.path.exists(path_from_root):
        raise FileNotFoundError(f"File not found: {file_name} in {path_from_root}")

    df = None
    if file_name.endswith(".csv"):
        df = pd.read_csv(path_from_root)
    elif file_name.endswith(".xlsx"):
        df = pd.read_excel(path_from_root)
    elif file_name.endswith(".json"):
        df = pd.read_json(path_from_root)
    elif file_name.endswith(".parquet"):
        df = pd.read_parquet(path_from_root)
    elif file_name.endswith(".feather"):
        df = pd.read_feather(path_from_root)
    else:
        raise ValueError(f"Unsupported file format: {file_name.split('.')[-1]}")

    if df is not None:
        print(f"DataFrame successfully loaded from '{file_name}'!")
        print(f"Loaded: {len(df)} rows and {len(df.columns)} columns.")

    # Save raw data to SQLite, used later for inference API
    try:
        conn = sqlite3.connect(DB_NAME)
        df.to_sql("raw_data", conn, if_exists="replace", index=True)
        conn.close()
        print(f"DataFrame successfully dumped to 'raw_data' table in '{DB_NAME.split('/')[-1]}'.")
    except Exception as e:
        print(f"Error dumping DataFrame to SQLite: {e}")
    return df


def transform_data(df: pd.DataFrame) -> pd.DataFrame:

    # During EDA, we found that the median house value is truncated to 500_001, double check:
    df.loc[df["median_house_value"] > 500_001, "median_house_value"] = 500_001
    # Same for housing_median_age:
    df.loc[df["housing_median_age"] > 52, "housing_median_age"] = 52
    # Fill missing values in total_bedrooms with the median value:
    df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())

    # Clustering location using DBSCAN:
    density_cluster = DBSCAN(eps=0.1, min_samples=5, metric="euclidean")
    clusters = density_cluster.fit_predict(df[["latitude", "longitude"]])
    df["clusters"] = clusters

    # Drop latitude and longitude columns:
    df.drop(columns=["latitude", "longitude"], axis=1, inplace=True)
    df["ocean_proximity"] = df["ocean_proximity"].map(ocean_proximity_map)
    # In case of missing or different values, assign the most common category (<1H OCEAN)
    df["ocean_proximity"] = df["ocean_proximity"].where(~df["ocean_proximity"].isin([1, 2, 3, 4, 5]), 1)

    return df

def load_data(df: pd.DataFrame, db_name: str, table_name: str) -> None:
    if not os.path.exists(db_name):
        db_name = os.path.join("..", db_name)
    try:
        conn = sqlite3.connect(db_name)
        df.to_sql(table_name, conn, if_exists="replace", index=True)
        conn.close()
        print(f"DataFrame successfully dumped to '{table_name}' table in '{db_name.split('/')[-1]}'.")
    except Exception as e:
        print(f"Error dumping DataFrame to SQLite: {e}")


async def etl_api(file_name: str, db_name: str, table_name: str) -> None:
    df = extract_data(file_name)
    df = transform_data(df)
    load_data(df, db_name, table_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETL Pipeline")
    parser.add_argument("file_name", type=str,
                        help="The file name, must include the extension (e.g., house_prices.csv) "
                             "and must in data folder")
    parser.add_argument("--db_name", type=str, default="etl_data.db",
                        help="The name of the SQLite database file.")
    parser.add_argument("--table_name", type=str, default="processed_data",
                        help="The name of the table to store the processed data.")
    args = parser.parse_args()

    df = extract_data(args.file_name)
    df = transform_data(df)
    load_data(df, args.db_name, args.table_name)
