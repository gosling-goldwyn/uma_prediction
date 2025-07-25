#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from time import sleep
import os
import sys
import argparse
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Add project root to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.prediction_utils.constants import PLACE_NUM
from scripts.prediction_utils.scraper import fetch_race_data_with_playwright

# --- Constants ---
DATASET_PATH = "data/dataset.parquet"
CACHE_DIR = "data/odds_cache"
OUTPUT_PATH = "data/odds_dataset.parquet"
ODDS_URL_TEMPLATE = "https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"


def generate_race_id(row):
    """Generate race_id from a row of the dataset."""
    try:
        year = str(row["year"])
        place_code = PLACE_NUM[row["place"]]
        kai = str(row["kai"]).zfill(2)
        day = str(row["day"]).zfill(2)
        race_num = str(row["race_num"]).zfill(2)
        return f"{year}{place_code}{kai}{day}{race_num}"
    except KeyError:
        return None


def load_proxies(proxy_file):
    """Load proxies from a file and ensure they have a scheme."""
    if not proxy_file or not os.path.exists(proxy_file):
        print("Proxy file not found or not provided. Continuing without proxies.")
        return None
    with open(proxy_file, "r") as f:
        proxies = []
        for line in f:
            line = line.strip()
            if line:
                if not line.startswith(
                    ("http://", "https://", "socks4://", "socks5://")
                ):
                    proxies.append(f"http://{line}")
                else:
                    proxies.append(line)

    if not proxies:
        print("Proxy file is empty. Continuing without proxies.")
        return None
    print(f"Loaded {len(proxies)} proxies.")
    return itertools.cycle(proxies)


def process_race(race_id, proxy):
    """
    Processes a single race: checks cache, scrapes if necessary, and returns odds data.
    """
    cache_path = os.path.join(CACHE_DIR, f"{race_id}.pkl")
    
    # 1. Check cache first
    if os.path.exists(cache_path):
        try:
            df_cached = pd.read_pickle(cache_path)
            # 2. Validate cached data
            if not (df_cached.empty or 'odds' not in df_cached.columns or df_cached['odds'].isnull().any()):
                # Return valid cached data
                return df_cached
        except Exception as e:
            print(f"Error reading cache file {cache_path}: {e}. Re-scraping...")

    # 3. Scrape if no valid cache exists
    url = ODDS_URL_TEMPLATE.format(race_id=race_id)
    df_detail, _ = fetch_race_data_with_playwright(url, proxy_server=proxy)

    if df_detail is not None and not df_detail.empty and "odds" in df_detail.columns and not df_detail['odds'].isnull().any():
        df_race_odds = df_detail[["horse_name", "odds"]].copy()
        df_race_odds["race_id"] = race_id
        df_race_odds.to_pickle(cache_path)  # Save to cache on success
        return df_race_odds
    
    return None # Return None on failure


def main(args):
    print("Initializing odds scraping...")
    os.makedirs(CACHE_DIR, exist_ok=True)

    proxy_cycler = load_proxies(args.proxy_file)

    try:
        df_dataset = pd.read_parquet(DATASET_PATH)
    except FileNotFoundError:
        print(f"Error: {DATASET_PATH} not found. Please run main_scraper first.")
        return

    df_races = df_dataset[["year", "place", "kai", "day", "race_num"]].drop_duplicates()
    df_races["race_id"] = df_races.apply(generate_race_id, axis=1)
    df_races.dropna(subset=["race_id"], inplace=True)

    all_odds_data = []
    race_ids = df_races["race_id"].tolist()
    total_races = len(race_ids)

    # --- Parallel Execution ---
    if proxy_cycler and args.parallel > 1:
        print(f"Running in parallel with {args.parallel} workers.")
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            # Create a future for each race
            futures = {executor.submit(process_race, race_id, next(proxy_cycler)): race_id for race_id in race_ids}
            
            # Process results as they complete
            for future in tqdm(as_completed(futures), total=total_races, desc="Scraping Races"):
                result_df = future.result()
                if result_df is not None and not result_df.empty:
                    all_odds_data.append(result_df)

    # --- Sequential Execution ---
    else:
        print("Running sequentially.")
        for race_id in tqdm(race_ids, desc="Scraping Races"):
            proxy = next(proxy_cycler) if proxy_cycler else None
            result_df = process_race(race_id, proxy)
            if result_df is not None and not result_df.empty:
                all_odds_data.append(result_df)
            
            # Adjust sleep time based on proxy usage
            if not proxy_cycler:
                sleep(5) # Be nice to the server if not using proxies

    print("Consolidating all odds data...")
    if all_odds_data:
        df_final = pd.concat(all_odds_data, ignore_index=True)
        df_final.to_parquet(OUTPUT_PATH)
        print(f"All odds data saved to {OUTPUT_PATH}")
    else:
        print("No odds data was collected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape odds data for horse races.")
    parser.add_argument(
        "--proxy-file",
        type=str,
        help="Path to a file containing a list of proxies (one per line).",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel processes to run for scraping. Requires a proxy file.",
    )
    args = parser.parse_args()
    main(args)
