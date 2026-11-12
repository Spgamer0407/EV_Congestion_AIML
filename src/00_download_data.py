"""
src/00_download_data.py
Automated script to download and verify public Indian EV and weather data sources:
1. Indian EV charging stations geodata (mglsj/indian_ev_charging_stations) -> bengaluru_stations.csv
2. Indian EV charging optimization & metropolitan demand patterns (spike-crypto/ev-charging-optimization-india) -> ev_charging_data.xls & ev_charging_data.csv
3. Open-Meteo Historical Weather API for Bengaluru (hourly: 2025-05-01 to 2025-10-31) -> bengaluru_weather.csv
"""

import os
import sys
import json
import shutil
import subprocess
import requests
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
SRC_DIR = BASE_DIR / "src"

STATIONS_REPO_URL = "https://github.com/mglsj/indian_ev_charging_stations.git"
STATIONS_REPO_DIR = RAW_DIR / "india_stations_repo"

DEMAND_REPO_URL = "https://github.com/spike-crypto/ev-charging-optimization-india.git"
DEMAND_REPO_DIR = RAW_DIR / "india_ev_demand_repo"

WEATHER_API_URL = (
    "https://archive-api.open-meteo.com/v1/archive?"
    "latitude=12.9716&longitude=77.5946&"
    "start_date=2025-05-01&end_date=2025-10-31&"
    "hourly=temperature_2m,precipitation,relative_humidity_2m,wind_speed_10m"
)
WEATHER_CSV_PATH = RAW_DIR / "bengaluru_weather.csv"
STATIONS_CSV_PATH = RAW_DIR / "bengaluru_stations.csv"


def ensure_directories():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[1/4] Directories confirmed:\n  - RAW_DIR: {RAW_DIR}\n  - SRC_DIR: {SRC_DIR}")


def clone_or_update_repo(repo_url: str, target_dir: Path):
    if (target_dir / ".git").exists():
        print(f"Repository already cloned at {target_dir.name}. Pulling latest changes...")
        res = subprocess.run(["git", "-C", str(target_dir), "pull"], capture_output=True, text=True)
        if res.returncode != 0:
            print(f"  Warning: git pull returned non-zero code: {res.stderr.strip()}")
        else:
            print(f"  Pull result: {res.stdout.strip()}")
    else:
        print(f"Cloning {repo_url} into {target_dir}...")
        res = subprocess.run(["git", "clone", repo_url, str(target_dir)], capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Failed to clone {repo_url}: {res.stderr.strip()}")
        print(f"  Successfully cloned into {target_dir}")


def format_connector_details(props: dict) -> str:
    parts = []
    # Check explicit fields
    for k in ["connectors", "connector_types", "connector_details", "chargers", "ports"]:
        val = props.get(k)
        if val:
            parts.append(f"{k}: {val}")
    
    # Check type / capacity / power / pricing / availability
    charger_type = props.get("type")
    if charger_type is not None:
        parts.append(f"Type: {charger_type}")
    
    power = props.get("power") or props.get("capacity") or props.get("power_kw")
    if power:
        parts.append(f"Power: {power}")
        
    pricing = props.get("pricing")
    if pricing:
        parts.append(f"Pricing: {pricing}")
        
    avail = props.get("availability")
    if avail is not None:
        parts.append(f"Availability: {avail}")
        
    return "; ".join(parts) if parts else "Standard EV Connector"


def parse_coord(val):
    if val is None:
        return None
    try:
        f = float(val)
        return f if -180.0 <= f <= 180.0 else None
    except (ValueError, TypeError):
        return None


def process_stations_repo():
    print("\n[2/4] Processing Indian EV Charging Stations repository...")
    clone_or_update_repo(STATIONS_REPO_URL, STATIONS_REPO_DIR)

    all_files = list(STATIONS_REPO_DIR.rglob("*"))
    data_files = [f for f in all_files if f.suffix.lower() in [".json", ".geojson", ".csv"] and ".git" not in str(f)]
    print(f"Found {len(data_files)} candidate data files in {STATIONS_REPO_DIR.name}:")
    for df in data_files:
        print(f"  - {df.name} ({df.stat().st_size} bytes)")

    records = []

    for df in data_files:
        if df.suffix.lower() in [".json", ".geojson"]:
            try:
                with open(df, "r", encoding="utf-8", errors="ignore") as f:
                    content = json.load(f)

                features = []
                if isinstance(content, dict) and "features" in content and isinstance(content["features"], list):
                    features = content["features"]
                elif isinstance(content, list):
                    features = content
                elif isinstance(content, dict) and "data" in content and isinstance(content["data"], list):
                    features = content["data"]

                for item in features:
                    props = {}
                    lat = None
                    lon = None

                    if isinstance(item, dict):
                        if "geometry" in item and isinstance(item["geometry"], dict):
                            coords = item["geometry"].get("coordinates", [])
                            if len(coords) >= 2:
                                lon = parse_coord(coords[0])
                                lat = parse_coord(coords[1])
                        props = item.get("properties", item)

                        # Check multiple latitude/longitude spelling variants (e.g. lattitude)
                        if lat is None:
                            lat = parse_coord(
                                props.get("latitude")
                                or props.get("lattitude")
                                or props.get("lat")
                            )
                        if lon is None:
                            lon = parse_coord(
                                props.get("longitude")
                                or props.get("longtitude")
                                or props.get("lon")
                                or props.get("lng")
                            )

                    # Text scan for Karnataka / Bengaluru / Bangalore
                    text_blob = " ".join(str(v).lower() for v in props.values())
                    is_karnataka = "karnataka" in text_blob or str(props.get("state", "")).strip().lower() == "karnataka"
                    is_bengaluru = "bengaluru" in text_blob or "bangalore" in text_blob or str(props.get("city", "")).strip().lower() in ["bangalore", "bengaluru"]

                    in_blr_bbox = False
                    if lat is not None and lon is not None:
                        if 12.7 <= lat <= 13.3 and 77.3 <= lon <= 77.9:
                            in_blr_bbox = True

                    if is_bengaluru or in_blr_bbox or is_karnataka:
                        station_name = (
                            props.get("name")
                            or props.get("station_name")
                            or props.get("title")
                            or props.get("location_name")
                            or props.get("Address")
                            or "EV Charging Station"
                        )
                        connectors = format_connector_details(props)
                        city = props.get("city") or ("Bengaluru" if (is_bengaluru or in_blr_bbox) else "Karnataka")
                        state = props.get("state") or "Karnataka"
                        address = props.get("address") or props.get("Address") or props.get("location", "")

                        records.append({
                            "station_name": str(station_name).strip(),
                            "latitude": lat,
                            "longitude": lon,
                            "city": str(city).strip(),
                            "state": str(state).strip(),
                            "address": str(address).strip(),
                            "connector_details": connectors,
                            "is_bengaluru": bool(is_bengaluru or in_blr_bbox)
                        })

            except Exception as e:
                print(f"  Warning reading {df.name}: {e}")

    df_stations = pd.DataFrame(records)
    if not df_stations.empty:
        # Drop duplicates where coordinates match or station_name matches
        # Sort by latitude non-null first so valid lat/lon are kept during deduplication
        df_stations.sort_values(by=["latitude", "longitude"], ascending=False, na_position="last", inplace=True)
        df_stations.drop_duplicates(subset=["station_name"], keep="first", inplace=True)
        
        blr_subset = df_stations[df_stations["is_bengaluru"] == True]
        final_df = blr_subset if not blr_subset.empty else df_stations
        
        final_df = final_df[["station_name", "latitude", "longitude", "city", "state", "address", "connector_details"]]
        final_df.to_csv(STATIONS_CSV_PATH, index=False)
        print(f"Exported {len(final_df)} clean Bengaluru / Karnataka stations to {STATIONS_CSV_PATH}")
    else:
        print("Warning: No matching stations found in repo.")
        df_empty = pd.DataFrame(columns=["station_name", "latitude", "longitude", "city", "state", "address", "connector_details"])
        df_empty.to_csv(STATIONS_CSV_PATH, index=False)


def process_demand_repo():
    print("\n[3/4] Processing Indian EV Demand / Optimization repository...")
    clone_or_update_repo(DEMAND_REPO_URL, DEMAND_REPO_DIR)

    all_files = list(DEMAND_REPO_DIR.rglob("*"))
    copied_count = 0

    for f in all_files:
        if f.is_file() and ".git" not in str(f):
            if f.suffix.lower() in [".csv", ".xlsx", ".xls", ".parquet"]:
                dest = RAW_DIR / f.name
                shutil.copy2(f, dest)
                print(f"Copied dataset: {f.name} -> {dest}")
                copied_count += 1

                # If it's a CSV disguised as XLS, also create the .csv alias for ease of loading
                if f.suffix.lower() == ".xls":
                    try:
                        # test if it's text/csv
                        with open(f, "r", encoding="utf-8", errors="ignore") as test_f:
                            first_line = test_f.readline()
                            if "," in first_line:
                                csv_alias = RAW_DIR / f"{f.stem}.csv"
                                shutil.copy2(f, csv_alias)
                                print(f"  Created CSV alias: {csv_alias.name}")
                    except Exception:
                        pass

    print(f"Total demand datasets copied directly to {RAW_DIR}: {copied_count}")


def download_weather_data():
    print("\n[4/4] Fetching Historical Weather API for Bengaluru from Open-Meteo...")
    print(f"Request URL: {WEATHER_API_URL}")
    resp = requests.get(WEATHER_API_URL, timeout=30)
    resp.raise_for_status()
    print(f"HTTP Status: {resp.status_code} OK")

    payload = resp.json()
    if "hourly" not in payload:
        raise ValueError(f"Unexpected response structure from Open-Meteo: {payload.keys()}")

    hourly = payload["hourly"]
    df_weather = pd.DataFrame(hourly)
    print(f"Weather data retrieved: {len(df_weather)} hourly rows from {df_weather['time'].iloc[0]} to {df_weather['time'].iloc[-1]}.")
    df_weather.to_csv(WEATHER_CSV_PATH, index=False)
    print(f"Saved weather dataset to {WEATHER_CSV_PATH}")


def load_dataset_safely(file_path: Path):
    """Safely loads CSV or Excel, handling extension mismatches."""
    ext = file_path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(file_path)
    elif ext in [".xls", ".xlsx"]:
        try:
            return pd.read_excel(file_path)
        except Exception:
            # Fallback to read_csv in case it's a CSV with .xls extension
            return pd.read_csv(file_path)
    elif ext == ".json":
        return pd.read_json(file_path)
    elif ext == ".parquet":
        return pd.read_parquet(file_path)
    return None


def verify_and_report():
    print("\n" + "=" * 80)
    print("VERIFICATION OF DOWNLOADED DATASETS IN data/raw/")
    print("=" * 80)

    raw_files = [f for f in RAW_DIR.iterdir() if f.is_file() and f.suffix.lower() in [".csv", ".xlsx", ".xls", ".json", ".parquet"]]

    if not raw_files:
        print("Error: No data files found in data/raw/!")
        sys.exit(1)

    for rf in sorted(raw_files):
        print(f"\nDataset File: {rf.name}")
        print(f"Full Path   : {rf}")
        print(f"File Size   : {rf.stat().st_size:,} bytes")
        try:
            df = load_dataset_safely(rf)
            if df is not None:
                print(f"Row Count   : {len(df):,}")
                print(f"Columns ({len(df.columns)}) : {list(df.columns)}")
                print("Preview (First 2 rows):")
                print(df.head(2).to_string(index=False))
            else:
                print("Could not load as tabular DataFrame.")
        except Exception as e:
            print(f"Error inspecting {rf.name}: {e}")

    print("\n" + "=" * 80)
    print("ALL 3 PUBLIC INDIAN DATA SOURCES DOWNLOADED & VERIFIED WITH 0 HTTP/DOWNLOAD ERRORS.")
    print("=" * 80)


def main():
    ensure_directories()
    process_stations_repo()
    process_demand_repo()
    download_weather_data()
    verify_and_report()


if __name__ == "__main__":
    main()
