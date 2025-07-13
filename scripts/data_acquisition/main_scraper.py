import os
import re
import pandas as pd
import json
from .scraping_util import fetch_page, parse_race_details

STATUS_FILE_PATH = "data/scraping_status.json"


def update_status(
    status: str,
    current_step: str = None,
    progress: int = None,
    total_pages: int = None,
    processed_pages: int = None,
    error: str = None,
):
    """スクレイピングの進捗状況をJSONファイルに更新します。

    Args:
        status (str): 現在のスクレイピングの状態（例: "idle", "running", "completed", "error"）。
        current_step (str, optional): 現在のステップの説明。
        progress (int, optional): 全体に対する進捗率（0-100）。
        total_pages (int, optional): 総ページ数。
        processed_pages (int, optional): 処理済みページ数。
        error (str, optional): エラーメッセージ。
    """
    try:
        with open(STATUS_FILE_PATH, "r+") as f:
            data = json.load(f)
            data["status"] = status
            if current_step is not None:
                data["current_step"] = current_step
            if progress is not None:
                data["progress"] = progress
            if total_pages is not None:
                data["total_pages"] = total_pages
            if processed_pages is not None:
                data["processed_pages"] = processed_pages
            if error is not None:
                data["error"] = error
            f.seek(0)
            json.dump(data, f, indent=4)
            f.truncate()
    except Exception as e:
        print(f"Error updating status file: {e}")


# 定数
URL_PREFIX = "https://db.netkeiba.com"
OVERVIEW_PICKLE_PATH = r"data\pandas_obj.pkl"
PROCESSED_URLS_PATH = r"data\processed_urls.txt"
CACHE_DIR = r"data\races"
OUTPUT_PARQUET_PATH = r"data\dataset.parquet"

# レースの回り順
COUNTERCLOCKWISE_PLACES = ["東京", "中京", "新潟"]
CLOCKWISE_PLACES = ["中山", "阪神", "京都", "札幌", "函館", "福島", "小倉"]


def load_processed_urls(path: str) -> set:
    """処理済みのURLのセットをファイルから読み込みます。

    Args:
        path (str): 処理済みURLが記録されたファイルのパス。

    Returns:
        set: 処理済みのURLのセット。
    """
    if not os.path.exists(path):
        return set()
    with open(path, "r") as f:
        return set(line.strip() for line in f)


def save_processed_url(path: str, url: str):
    """処理済みのURLをファイルに追記します。

    Args:
        path (str): 処理済みURLを記録するファイルのパス。
        url (str): 保存するURL。
    """
    with open(path, "a") as f:
        f.write(url + "\n")


def get_race_turn(place: str) -> str:
    """開催地からレースの回り（右・左）を返します。

    Args:
        place (str): 開催地名。

    Returns:
        str: '右'、'左'、または'不明'。
    """
    if place in CLOCKWISE_PLACES:
        return "右"
    elif place in COUNTERCLOCKWISE_PLACES:
        return "左"
    else:
        return "不明"


def process_horse_weight(weight_str: str) -> tuple:
    """馬体重の文字列を体重と増減に分割します。

    Args:
        weight_str (str): 馬体重の文字列 (例: '480(+2)')。

    Returns:
        tuple: (馬体重, 体重増減)。解析できない場合は (-1, 0)。
    """
    try:
        weight = float(weight_str[: weight_str.find("(")])
        change = float(weight_str[weight_str.find("(") :].strip("()"))
        return weight, change
    except (ValueError, IndexError):
        return -1, 0


def main():
    """スクレイピング処理のメイン関数。"""
    update_status(
        status="running", current_step="Scraping Race Detail Data", progress=0
    )
    os.makedirs(CACHE_DIR, exist_ok=True)
    processed_urls = load_processed_urls(PROCESSED_URLS_PATH)

    try:
        df_overview = pd.read_pickle(OVERVIEW_PICKLE_PATH)
    except FileNotFoundError:
        print(f"Error: {OVERVIEW_PICKLE_PATH} not found.")
        update_status(status="error", error=f"Error: {OVERVIEW_PICKLE_PATH} not found.")
        return

    all_race_data = []
    total_races = len(df_overview)
    processed_races_count = 0

    try:
        for i, overview_row in df_overview.iterrows():
            race_url_suffix = overview_row["レース名"]
            race_url = URL_PREFIX + race_url_suffix

            if race_url in processed_urls:
                print(f"Skipping already processed: {race_url}")
                processed_races_count += 1
                continue

            if overview_row["開催日"] == "開催日":
                processed_races_count += 1
                update_status(
                    status="running",
                    current_step=f"Skipping invalid row: {overview_row['開催日']}",
                    processed_pages=processed_races_count,
                    progress=int(processed_races_count / total_races * 100),
                )
                continue

            # --- レース概要情報の処理 ---
            place = re.sub(r"^\d+", "", re.sub(r"\d+", "", overview_row["開催"]))
            base_race_info = {
                "year": int(overview_row["開催日"].split("/")[0]),
                "date": "/ ".join(overview_row["開催日"].split("/")[1:]),
                "month": int(overview_row["開催日"].split("/")[1]),
                "race_num": int(overview_row["R"]),
                "field": re.sub(r"[0-9]+", "", overview_row["距離"]),
                "dist": int(re.sub(r"\D", "", overview_row["距離"])),
                "turn": get_race_turn(place),
                "weather": overview_row["天気"],
                "field_cond": overview_row["馬場"],
                "kai": int(re.sub(r"\D+\d+", "", overview_row["開催"])),
                "day": int(re.sub(r"\d+\D+", "", overview_row["開催"])),
                "place": place,
            }

            # --- レース詳細情報の取得 ---
            detail_cache_path = os.path.join(
                CACHE_DIR, race_url_suffix.replace("/", "_") + ".pkl"
            )
            if os.path.exists(detail_cache_path):
                df_detail = pd.read_pickle(detail_cache_path)
            else:
                soup = fetch_page(race_url)
                if soup is None:
                    processed_races_count += 1
                    update_status(
                        status="running",
                        current_step=f"Failed to fetch {race_url}",
                        processed_pages=processed_races_count,
                        progress=int(processed_races_count / total_races * 100),
                    )
                    continue  # ページ取得失敗
                df_detail = parse_race_details(soup)
                if df_detail.empty:
                    print(f"Could not parse details for: {race_url}")
                    processed_races_count += 1
                    update_status(
                        status="running",
                        current_step=f"Could not parse details for: {race_url}",
                        processed_pages=processed_races_count,
                        progress=int(processed_races_count / total_races * 100),
                    )
                    continue  # 解析失敗
                df_detail.to_pickle(detail_cache_path)

            base_race_info["sum_num"] = len(df_detail)

            # --- 馬ごとの詳細情報をマージ ---
            for _, detail_row in df_detail.iterrows():
                horse_weight, weight_change = process_horse_weight(detail_row["馬体重"])

                full_race_info = base_race_info.copy()
                full_race_info.update(
                    {
                        "prize": float(detail_row["賞金(万円)"].replace(",", ""))
                        if detail_row["賞金(万円)"]
                        else 0,
                        "rank": int(detail_row["着順"])
                        if detail_row["着順"].isdigit()
                        else -1,
                        "horse_num": int(detail_row["馬番"]),
                        "horse_name": detail_row["馬名"],
                        "sex": re.sub(r"[0-9]+", "", detail_row["性齢"]),
                        "age": int(re.sub(r"\D", "", detail_row["性齢"])),
                        "weight_carry": float(detail_row["斤量"]),
                        "horse_weight": horse_weight,
                        "weight_change": weight_change,
                        "jockey": detail_row["騎手"],
                        "time": detail_row["タイム"],
                        "l_days": "",  # l_daysは元々空だったので維持
                    }
                )
                all_race_data.append(list(full_race_info.values()))

            save_processed_url(PROCESSED_URLS_PATH, race_url)
            print(f"Successfully processed: {race_url}")
            processed_races_count += 1
            update_status(
                status="running",
                current_step=f"Processed {race_url}",
                processed_pages=processed_races_count,
                total_pages=total_races,
                progress=int(processed_races_count / total_races * 100),
            )

        if not all_race_data:
            print("No new data scraped.")
            update_status(status="completed", current_step="No new data scraped.")
            return

        df_all = pd.DataFrame(all_race_data, columns=list(full_race_info.keys()))
        df_all.to_parquet(OUTPUT_PARQUET_PATH)
        print(f"All data saved to {OUTPUT_PARQUET_PATH}")
        update_status(
            status="completed",
            current_step="Race detail data scraped and saved.",
            progress=100,
        )

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        update_status(status="error", error=str(e))


if __name__ == "__main__":
    main()
