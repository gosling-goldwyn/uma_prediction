import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import os
from time import sleep
import re
import json
from collections import defaultdict

# --- Constants and Configuration ---
STATUS_FILE_PATH = "data/scraping_status.json"
DATASET_PARQUET_PATH = "data/dataset.parquet"
SPEED_INDEX_CACHE_DIR = "data/speed_index_cache"  # キャッシュディレクトリ名を変更
OUTPUT_PARQUET_PATH = "data/index_dataset.parquet"
SPEED_INDEX_URL_PREFIX = "http://jiro8.sakura.ne.jp/index.php?code="

# レースデータと対応をとるためのキー
COMMON_HEAD = [
    "year",
    "date",
    "month",
    "kai",
    "day",
    "place",
    "sum_num",
    "horse_num",
    "horse_name",
]
# スピード指数
INDEX_HEAD = [
    "lead_idx",
    "1st_lead_idx",
    "2nd_lead_idx",
    "3rd_lead_idx",
    "4th_lead_idx",
    "5th_lead_idx",
    "pace_idx",
    "1st_pace_idx",
    "2nd_pace_idx",
    "3rd_pace_idx",
    "4th_pace_idx",
    "5th_pace_idx",
    "rising_idx",
    "1st_rising_idx",
    "2nd_rising_idx",
    "3rd_rising_idx",
    "4th_rising_idx",
    "5th_rising_idx",
    "speed_idx",
    "1st_speed_idx",
    "2nd_speed_idx",
    "3rd_speed_idx",
    "4th_speed_idx",
    "5th_speed_idx",
]

# 前走データカラム
PREV_RACE_HEAD = [
    "1st_place",
    "1st_weather",
    "2nd_place",
    "2nd_weather",
    "3rd_place",
    "3rd_weather",
    "4th_place",
    "4th_weather",
    "5th_place",
    "5th_weather",
    "1st_field",
    "1st_dist",
    "1st_condi",
    "2nd_field",
    "2nd_dist",
    "2nd_condi",
    "3rd_field",
    "3rd_dist",
    "3rd_condi",
    "4th_field",
    "4th_dist",
    "4th_condi",
    "5th_field",
    "5th_dist",
    "5th_condi",
    "1st_sum_num",
    "1st_horse_num",
    "2nd_sum_num",
    "2nd_horse_num",
    "3rd_sum_num",
    "3rd_horse_num",
    "4th_sum_num",
    "4th_horse_num",
    "5th_sum_num",
    "5th_horse_num",
    "1st_rank",
    "2nd_rank",
    "3rd_rank",
    "4th_rank",
    "5th_rank",
]

# 最終的な結合データフレームのカラム
FINAL_COLS = (
    [
        "year",
        "date",
        "month",
        "race_num",
        "field",
        "dist",
        "turn",
        "weather",
        "field_cond",
        "kai",
        "day",
        "place",
        "sum_num",
        "prize",
        "rank",
        "horse_num",
        "horse_name",
        "sex",
        "age",
        "weight_carry",
        "horse_weight",
        "weight_change",
        "jockey",
        "time",
        "l_days",
    ]
    + INDEX_HEAD
    + PREV_RACE_HEAD
)

# 地方競馬場と対応する数値
PLACE_NUM = {
    "札幌": "01",
    "函館": "02",
    "福島": "03",
    "新潟": "04",
    "東京": "05",
    "中山": "06",
    "中京": "07",
    "京都": "08",
    "阪神": "09",
    "小倉": "10",
}

REPLACE_DICT = {
    "レース名": "前走の成績.レース名",
    "コース": "前走の成績.コース",
    "騎手,斤量": "前走の成績.騎手,斤量",
    "頭数,馬番,人気": "前走の成績.頭数,馬番,人気",
    "タイム,(着順)": "前走の成績.タイム,(着順)",
    "ﾍﾟｰｽ,脚質,上3F": "前走の成績.ﾍﾟｰｽ,脚質,上3F",
    "通過順位": "前走の成績.通過順位",
    "ﾄｯﾌﾟ(ﾀｲﾑ差)": "前走の成績.ﾄｯﾌﾟ(ﾀｲﾑ差)",
    "馬体重()3F順": "前走の成績.馬体重()3F順",
    "先行指数": "前走の成績.先行指数",
    "ペース指数": "前走の成績.ペース指数",
    "上がり指数": "前走の成績.上がり指数",
    "スピード指数": "前走の成績.スピード指数",
    "レース結果.1": "レース結果.ﾍﾟｰｽ,脚質,上3F",
    "レース結果.2": "レース結果.通過順位",
    "レース結果.3": "レース結果.馬体重",
    "レース結果.4": "レース結果.先行指数",
    "レース結果.5": "レース結果.ペース指数",
    "レース結果.6": "レース結果.上がり指数",
    "レース結果.7": "レース結果.スピード指数",
    "２走前の成績.1": "２走前の成績.レース名",
    "２走前の成績.2": "２走前の成績.コース",
    "２走前の成績.3": "２走前の成績.騎手,斤量",
    "２走前の成績.4": "２走前の成績.頭数,馬番,人気",
    "２走前の成績.5": "２走前の成績.タイム,(着順)",
    "２走前の成績.6": "２走前の成績.ﾍﾟｰｽ,脚質,上3F",
    "２走前の成績.7": "２走前の成績.通過順位",
    "２走前の成績.8": "２走前の成績.ﾄｯﾌﾟ(ﾀｲﾑ差)",
    "２走前の成績.9": "２走前の成績.馬体重()3F順",
    "２走前の成績.10": "２走前の成績.先行指数",
    "２走前の成績.11": "２走前の成績.ペース指数",
    "２走前の成績.12": "２走前の成績.上がり指数",
    "２走前の成績.13": "２走前の成績.スピード指数",
    "３走前の成績.1": "３走前の成績.レース名",
    "３走前の成績.2": "３走前の成績.コース",
    "３走前の成績.3": "３走前の成績.騎手,斤量",
    "３走前の成績.4": "３走前の成績.頭数,馬番,人気",
    "３走前の成績.5": "３走前の成績.タイム,(着順)",
    "３走前の成績.6": "３走前の成績.ﾍﾟｰｽ,脚質,上3F",
    "３走前の成績.7": "３走前の成績.通過順位",
    "３走前の成績.8": "３走前の成績.ﾄｯﾌﾟ(ﾀｲﾑ差)",
    "３走前の成績.9": "３走前の成績.馬体重()3F順",
    "３走前の成績.10": "３走前の成績.先行指数",
    "３走前の成績.11": "３走前の成績.ペース指数",
    "３走前の成績.12": "３走前の成績.上がり指数",
    "３走前の成績.13": "３走前の成績.スピード指数",
    "４走前の成績.1": "４走前の成績.レース名",
    "４走前の成績.2": "４走前の成績.コース",
    "４走前の成績.3": "４走前の成績.騎手,斤量",
    "４走前の成績.4": "４走前の成績.頭数,馬番,人気",
    "４走前の成績.5": "４走前の成績.タイム,(着順)",
    "４走前の成績.6": "４走前の成績.ﾍﾟｰｽ,脚質,上3F",
    "４走前の成績.7": "４走前の成績.通過順位",
    "４走前の成績.8": "４走前の成績.ﾄｯﾌﾟ(ﾀｲﾑ差)",
    "４走前の成績.9": "４走前の成績.馬体重()3F順",
    "４走前の成績.10": "４走前の成績.先行指数",
    "４走前の成績.11": "４走前の成績.ペース指数",
    "４走前の成績.12": "４走前の成績.上がり指数",
    "４走前の成績.13": "４走前の成績.スピード指数",
    "５走前の成績.1": "５走前の成績.レース名",
    "５走前の成績.2": "５走前の成績.コース",
    "５走前の成績.3": "５走前の成績.騎手,斤量",
    "５走前の成績.4": "５走前の成績.頭数,馬番,人気",
    "５走前の成績.5": "５走前の成績.タイム,(着順)",
    "５走前の成績.6": "５走前の成績.ﾍﾟｰｽ,脚質,上3F",
    "５走前の成績.7": "５走前の成績.通過順位",
    "５走前の成績.8": "５走前の成績.ﾄｯﾌﾟ(ﾀｲﾑ差)",
    "５走前の成績.9": "５走前の成績.馬体重()3F順",
    "５走前の成績.10": "５走前の成績.先行指数",
    "５走前の成績.11": "５走前の成績.ペース指数",
    "５走前の成績.12": "５走前の成績.上がり指数",
    "５走前の成績.13": "５走前の成績.スピード指数",
    "レース結果.ﾍﾟｰｽ,脚質,上3F": "ﾍﾟｰｽ,脚質,上3F",
    "レース結果.通過順位": "通過順位",
    "レース結果.馬体重": "馬体重",
    "レース結果.先行指数": "先行指数",
    "レース結果.ペース指数": "ペース指数",
    "レース結果.上がり指数": "上がり指数",
    "レース結果.スピード指数": "スピード指数",
}


# --- Utility Functions ---
def update_status(
    status: str,
    current_step: str = None,
    progress: int = None,
    total_pages: int = None,
    processed_pages: int = None,
    error: str = None,
):
    """スクレイピングの進捗状況をJSONファイルに更新します。"""
    try:
        if not os.path.exists(STATUS_FILE_PATH):
            # ファイルが存在しない場合は初期化
            initial_status = {
                "status": "idle",
                "current_step": "Initializing...",
                "progress": 0,
                "total_pages": 0,
                "processed_pages": 0,
                "error": "None",
            }
            with open(STATUS_FILE_PATH, "w") as f:
                json.dump(initial_status, f, indent=4)

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


def get_value_or_empty(arr):
    """配列の最初の要素を返すか、空の場合は空文字列を返します。"""
    return arr[0] if arr.size > 0 else ""


def parse_speed_index_table(soup: BeautifulSoup, num_horses: int) -> pd.DataFrame:
    """スピード指数ページのテーブルをパースします。"""
    race_result_table_base = soup.find_all("table", attrs={"class": "c1"})
    if not race_result_table_base:
        return pd.DataFrame()

    # 最初のテーブルが目的のテーブルと仮定
    table = race_result_table_base[0]

    # ヘッダーとデータを抽出
    headers = []
    data_rows = []

    # データ行の処理
    # 各行のtd要素からデータを抽出
    for tr in table.find_all("tr"):
        tds = tr.find_all("td", recursive=False)
        if len(tds) == num_horses + 1:  # 列数が出馬数+1ならデータ行と判断
            row_data = [
                td.text.strip() for td in reversed(tds)
            ]  # 逆順になっているので反転
            data_rows.append(row_data[1:])
            if row_data[1].find("着") >= 0:
                headers.append("着順")
            elif (
                "レース結果" in headers
                and "２走前の成績" not in headers
                and row_data[0] == ""
            ):
                headers.append("レース結果")
            elif (
                "２走前の成績" in headers
                and "３走前の成績" not in headers
                and row_data[0] == ""
            ):
                headers.append("２走前の成績")
            elif (
                "３走前の成績" in headers
                and "４走前の成績" not in headers
                and row_data[0] == ""
            ):
                headers.append("３走前の成績")
            elif (
                "４走前の成績" in headers
                and "５走前の成績" not in headers
                and row_data[0] == ""
            ):
                headers.append("４走前の成績")
            elif "５走前の成績" in headers and row_data[0] == "":
                headers.append("５走前の成績")
            else:
                headers.append(row_data[0])

    if not data_rows:
        return pd.DataFrame()

    # DataFrameに変換
    df = pd.DataFrame(data_rows).T
    df.columns = headers
    df["馬名"] = df["馬名"].str.replace("ｌ", "ー")
    df = make_columns_unique(df.copy())
    for k, v in REPLACE_DICT.items():
        if k in df.columns:
            df.rename(columns={k: v}, inplace=True)

    return df


def parse_previous_race_data(df_detail_row: pd.Series) -> dict:
    """前走データをパースして辞書形式で返します。"""
    parsed_data = {}

    # 成績 (場所, 天気)
    for i, col_name in enumerate(
        ["前走の成績", "２走前の成績", "３走前の成績", "４走前の成績", "５走前の成績"]
    ):
        if col_name in df_detail_row and df_detail_row[col_name]:
            val = df_detail_row[col_name]
            place_match = re.sub(r"\d+/\d+|\D$", "", val)  # 場所
            weather_match = re.sub(r"\d+/\d+\D", "", val)  # 天気
            parsed_data[f"{i + 1}st_place"] = place_match
            parsed_data[f"{i + 1}st_weather"] = weather_match
        else:
            parsed_data[f"{i + 1}st_place"] = ""
            parsed_data[f"{i + 1}st_weather"] = ""

    # コース (芝/ダート, 距離, 馬場状態)
    for i, col_name in enumerate(
        [
            "前走の成績.コース",
            "２走前の成績.コース",
            "３走前の成績.コース",
            "４走前の成績.コース",
            "５走前の成績.コース",
        ]
    ):  # 複数ある可能性
        if col_name in df_detail_row and df_detail_row[col_name]:
            val = df_detail_row[col_name]
            field_match = re.sub(r"\d+\D", "", val)  # 芝 or ダート
            dist_match = -1
            try:
                dist_match = int(re.sub(r"^\D|\D$", "", val))  # 距離
            except ValueError:
                pass  # 変換できない場合は-1のまま
            condi_match = re.sub(r"\D\d+", "", val)  # 馬場状態

            parsed_data[f"{i + 1}st_field"] = field_match
            parsed_data[f"{i + 1}st_dist"] = dist_match
            parsed_data[f"{i + 1}st_condi"] = condi_match
        else:
            parsed_data[f"{i + 1}st_field"] = ""
            parsed_data[f"{i + 1}st_dist"] = -1
            parsed_data[f"{i + 1}st_condi"] = ""

    # 頭数, 馬番, 人気
    for i, col_name in enumerate(
        [
            "前走の成績.頭数,馬番,人気",
            "２走前の成績.頭数,馬番,人気",
            "３走前の成績.頭数,馬番,人気",
            "４走前の成績.頭数,馬番,人気",
            "５走前の成績.頭数,馬番,人気",
        ]
    ):
        if col_name in df_detail_row and df_detail_row[col_name]:
            val = df_detail_row[col_name]
            sum_num = -1
            horse_num = -1
            try:
                sum_num = int(re.sub(r"ﾄ.+", "", val))  # 馬数
                horse_num = int(re.sub(r"\d+ﾄ|番.*", "", val))  # 馬番
            except ValueError:
                pass
            parsed_data[f"{i + 1}st_sum_num"] = sum_num
            parsed_data[f"{i + 1}st_horse_num"] = horse_num
        else:
            parsed_data[f"{i + 1}st_sum_num"] = -1
            parsed_data[f"{i + 1}st_horse_num"] = -1

    # タイム,(着順)
    for i, col_name in enumerate(
        [
            "前走の成績.タイム,(着順)",
            "２走前の成績.タイム,(着順)",
            "３走前の成績.タイム,(着順)",
            "４走前の成績.タイム,(着順)",
            "５走前の成績.タイム,(着順)",
        ]
    ):
        if col_name in df_detail_row and df_detail_row[col_name]:
            val = df_detail_row[col_name]
            rank = -1
            try:
                rank_char = re.sub(r"\d\.\d\d\.\d", "", val)
                # 全角数字の着順を変換するロジック (例: １ -> 1)
                rank_char = rank_char.translate(
                    str.maketrans("０１２３４５６７８９", "0123456789")
                )
                rank_char = rank_char.translate(str.maketrans("①②③④⑤⑥⑦⑧⑨", "123456789"))
                rank_char = rank_char.translate(
                    str.maketrans(
                        {
                            "⑩": "10",
                            "⑪": "11",
                            "⑫": "12",
                            "⑬": "13",
                            "⑭": "14",
                            "⑮": "15",
                            "⑯": "16",
                            "⑰": "17",
                            "⑱": "18",
                            "⑲": "19",
                            "⑳": "20",
                        }
                    )
                )
                rank = int(rank_char)
            except ValueError:
                pass
            parsed_data[f"{i + 1}st_rank"] = rank
        else:
            parsed_data[f"{i + 1}st_rank"] = -1

    return parsed_data


def make_columns_unique(df):
    """
    DataFrameの重複する列名を連番を付与して一意にします。
    例: 'A', 'B', 'A', 'C', 'B' -> 'A', 'B', 'A.1', 'C', 'B.1'
    """
    cols = df.columns
    seen = defaultdict(int)  # 各列名が出現した回数を記録
    new_cols = []

    for col in cols:
        if seen[col] > 0:
            # 2回目以降の出現の場合、連番を付与
            new_cols.append(f"{col}.{seen[col]}")
        else:
            # 初めての出現の場合、そのままの名前を使用
            new_cols.append(col)
        seen[col] += 1  # 出現回数をインクリメント
    df.columns = new_cols  # 新しい列名をDataFrameに設定
    return df


# --- Main Logic ---
def main():
    update_status(
        status="running",
        current_step="Starting Speed Index Scraping Workflow",
        progress=0,
    )
    os.makedirs(SPEED_INDEX_CACHE_DIR, exist_ok=True)

    try:
        df_allrace = pd.read_parquet(DATASET_PARQUET_PATH)
    except FileNotFoundError:
        print(
            f"Error: {DATASET_PARQUET_PATH} not found. Please run main_scraper first."
        )
        update_status(status="error", error=f"{DATASET_PARQUET_PATH} not found.")
        return

    all_merged_data = []
    total_races = len(df_allrace)

    for i, race_row in df_allrace.iterrows():
        current_progress = int((i / total_races) * 100)
        update_status(
            status="running",
            current_step=f"Processing race {i + 1}/{total_races}: {race_row['horse_name']}",
            progress=current_progress,
            total_pages=total_races,
            processed_pages=i,
        )

        # URLパラメータの生成
        try:
            param = (
                str(race_row["year"])[-2:]
                + PLACE_NUM[race_row["place"]]
                + str(race_row["kai"]).zfill(2)
                + str(race_row["day"]).zfill(2)
                + str(race_row["race_num"]).zfill(2)
            )
        except KeyError:
            print(f"Skipping race due to unknown place: {race_row['place']}")
            continue
        except Exception as e:
            print(f"Skipping race due to URL parameter error: {e} in row {i}")
            continue

        cache_path = os.path.join(SPEED_INDEX_CACHE_DIR, param + ".pkl")
        df_speed_index = pd.DataFrame()

        if os.path.exists(cache_path):
            try:
                df_speed_index = pd.read_pickle(cache_path)
            except Exception as e:
                print(f"Error loading cache {cache_path}: {e}. Re-scraping.")
                os.remove(cache_path)  # 破損したキャッシュを削除

        if df_speed_index.empty:  # キャッシュがないか、読み込みに失敗した場合
            print(f"Scraping speed index for: {param}")
            sleep(1)  # サーバー負荷軽減

            try:
                res = requests.get(SPEED_INDEX_URL_PREFIX + param)
                res.raise_for_status()  # HTTPエラーがあれば例外を発生
                res.encoding = "cp932"  # Shift_JISの拡張であるcp932に設定
                soup = BeautifulSoup(res.text, "html.parser")
            except requests.exceptions.RequestException as e:
                print(f"Error fetching {SPEED_INDEX_URL_PREFIX + param}: {e}")
                soup = None

            if soup:
                df_speed_index = parse_speed_index_table(soup, race_row["sum_num"])
                if not df_speed_index.empty:
                    df_speed_index.to_pickle(cache_path)
                else:
                    print(f"Warning: No speed index data parsed for {param}")
            else:
                print(f"Error: Failed to fetch page for {param}")
                continue  # ページ取得失敗

        if df_speed_index.empty:
            print(f"Skipping race {param} due to empty speed index data.")
            continue

        # 現在の馬のデータに絞り込み
        # 馬名が完全に一致する行、または部分一致で末尾が一致する行を検索
        if "馬名" in df_speed_index.columns:
            horse_speed_data = df_speed_index[
                df_speed_index["馬名"].str.contains(race_row["horse_name"], na=False)
            ]
        else:
            print(
                f"Error: '馬名' column not found in df_speed_index for {param}. Skipping."
            )
            continue

        if horse_speed_data.empty:
            print(
                f"Warning: Speed index data for horse '{race_row['horse_name']}' not found in {param}. Skipping."
            )
            continue

        # 複数行見つかった場合は最初の行を使用 (同名馬対策)
        horse_speed_data = horse_speed_data.iloc[0]

        # スピード指数を抽出
        extracted_index_data = {}
        for col_name in INDEX_HEAD:
            # 元のコードではset_axisでカラム名を変更していたが、ここでは直接抽出
            # 例: 'lead_idx'は'先行指数'に対応すると仮定
            original_col_name = (
                col_name.replace("_idx", "指数")
                .replace("1st_", "前走の成績.")
                .replace("2nd_", "２走前の成績.")
                .replace("3rd_", "３走前の成績.")
                .replace("4th_", "４走前の成績.")
                .replace("5th_", "５走前の成績.")
                .replace("pace", "ペース")
                .replace("rising", "上がり")
                .replace("speed", "スピード")
                .replace("lead", "先行")
            )

            if original_col_name in horse_speed_data:
                val = horse_speed_data[original_col_name]
                # 数値に変換できるものは変換、できないものはNaN
                try:
                    extracted_index_data[col_name] = float(val)
                except (ValueError, TypeError):
                    extracted_index_data[col_name] = np.nan
            else:
                extracted_index_data[col_name] = np.nan  # カラムがない場合はNaN

        # 前走データをパース
        parsed_previous_race_data = parse_previous_race_data(horse_speed_data)

        # 全データを結合
        combined_data = race_row.to_dict()
        combined_data.update(extracted_index_data)
        combined_data.update(parsed_previous_race_data)

        # 最終的なカラムリストに合わせてデータを整形
        row_for_df = {}
        for col in FINAL_COLS:
            row_for_df[col] = combined_data.get(
                col, np.nan
            )  # 存在しないカラムはNaNで埋める

        all_merged_data.append(row_for_df)

    if not all_merged_data:
        print("No data processed for speed index.")
        update_status(status="completed", current_step="No speed index data processed.")
        return

    df_final = pd.DataFrame(all_merged_data, columns=FINAL_COLS)
    df_final.to_parquet(OUTPUT_PARQUET_PATH)
    print(f"All speed index data saved to {OUTPUT_PARQUET_PATH}")
    update_status(
        status="completed",
        current_step="Speed index data scraped and saved.",
        progress=100,
    )


if __name__ == "__main__":
    main()
