import os
import re
import pandas as pd
import json
import requests
from bs4 import BeautifulSoup
import numpy as np
import joblib
import tensorflow as tf
import sys

# GPUが利用可能か確認し、設定を行う
physical_devices = tf.config.list_physical_devices("GPU")
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU is available and memory growth is enabled.")
    except:
        print("Could not set GPU memory growth.")
else:
    print("No GPU devices found. Using CPU.")

# --- Constants ---
URL_PREFIX = "https://db.netkeiba.com"
DATA_FILE = "data/index_dataset.parquet"  # parquetファイルへのパス

from scripts.model_training.train import get_model_path

# レースの回り順
COUNTERCLOCKWISE_PLACES = ["東京", "中京", "新潟"]
CLOCKWISE_PLACES = ["中山", "阪神", "京都", "札幌", "函館", "福島", "小倉"]

# 地方競馬場と対応する数値 (jiro8は使わないが、過去データの日付生成に必要かもしれないので残す)
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

# スピード指数関連のカラム (parquetから取得する)
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

# 前走データカラム (parquetから取得する)
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


# --- Utility Functions ---
def fetch_page(url: str):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        res = requests.get(url, headers=headers)
        res.raise_for_status()
        res.encoding = "EUC-JP"
        return BeautifulSoup(res.text, "html.parser")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None


def parse_race_details(soup: BeautifulSoup) -> pd.DataFrame:
    table = soup.find("table", class_="Shutuba_Table")
    if not table:
        return pd.DataFrame()
    df = pd.read_html(str(table))[0]
    # カラム名をフラット化し、不要な空白を削除
    # MultiIndexの場合、タプルの最後の要素を使用
    df.columns = [col[-1].strip() if isinstance(col, tuple) else col.strip() for col in df.columns.values]

    # '馬 番'カラムを特定し、数値のみの行をフィルタリング
    horse_num_col_candidates = [
        col for col in df.columns if "馬" in col and "番" in col
    ]
    if not horse_num_col_candidates:
        print("Error: '馬 番' column not found in parsed table.")
        return pd.DataFrame()
    horse_num_col = horse_num_col_candidates[0]

    df = df[df[horse_num_col].astype(str).str.isdigit()]

    # 性齢カラムの特定
    sex_age_col_candidates = [col for col in df.columns if "性齢" in col]
    if not sex_age_col_candidates:
        print("Error: '性齢' column not found in parsed table.")
        return pd.DataFrame()
    sex_age_col = sex_age_col_candidates[0]

    df.rename(
        columns={
            "枠": "waku",
            horse_num_col: "horse_num",
            "印": "check_mark",
            sex_age_col: "sex_age",
            "斤量": "weight_carry",
            "騎手": "jockey",
            "厩舎": "stable",
            "馬体重 (増減)": "horse_weight_change",
            "人気": "popularity",
            "Unnamed: 9_level_1": "odds", # Explicitly rename the odds column
        },
        inplace=True,
    )
    # 馬名カラムは既に正しく設定されているはずなので、ここではリネームしない
    # df.rename(columns={"馬名": "horse_name"}, inplace=True) # 削除またはコメントアウト

    # The odds column is now explicitly renamed, so this logic is no longer needed.
    # Keeping it commented out for reference.
    # odds_col_candidates = [
    #     col for col in df.columns if "オッズ" in col or "人気" in col
    # ]
    # if len(odds_col_candidates) >= 2:
    #     popularity_col_idx = -1
    #     for i, col in enumerate(df.columns):
    #         if "人気" in col:
    #             popularity_col_idx = i
    #             break

    #     if popularity_col_idx > 0:
    #         odds_col = df.columns[popularity_col_idx - 1]
    #         df.rename(columns={odds_col: "odds"}, inplace=True)
    #     else:
    #         print(
    #             "Warning: Could not reliably find 'オッズ' column based on '人気' position. Setting to default."
    #         )
    #         df["odds"] = np.nan
    # else:
    #     print(
    #         "Warning: Not enough 'オッズ' or '人気' related columns found. Setting 'odds' to default."
    #     )
    #     df["odds"] = np.nan

    return df


def get_race_turn(place: str) -> str:
    if place in CLOCKWISE_PLACES:
        return "右"
    elif place in COUNTERCLOCKWISE_PLACES:
        return "左"
    else:
        return "不明"


def process_horse_weight(weight_str: str) -> tuple:
    if not isinstance(weight_str, str) or weight_str == "---":
        return -1, 0
    match = re.match(r"(\d+)\((\+?-?\d+)\)", weight_str)
    if match:
        return int(match.group(1)), int(match.group(2))
    elif weight_str.isdigit():
        return int(weight_str), 0
    return -1, 0


# --- Model Prediction Functions ---
def preprocess_data_for_prediction(df, model_type="cnn", flat_features_columns=None, imputation_values=None):
    if model_type == "cnn":
        sequence_features = [
            "1st_speed_idx",
            "2nd_speed_idx",
            "3rd_speed_idx",
            "4th_speed_idx",
            "5th_speed_idx",
            "1st_lead_idx",
            "2nd_lead_idx",
            "3rd_lead_idx",
            "4th_lead_idx",
            "5th_lead_idx",
            "1st_pace_idx",
            "2nd_pace_idx",
            "3rd_pace_idx",
            "4th_pace_idx",
            "5th_pace_idx",
            "1st_rising_idx",
            "2nd_rising_idx",
            "3rd_rising_idx",
            "4th_rising_idx",
            "5th_rising_idx",
        ]
        X_seq_df = df[sequence_features].copy()
        for col in sequence_features:
            # Use imputation_values if available, otherwise fallback to 0
            fill_value = imputation_values.get(col, 0) if imputation_values else 0
            X_seq_df[col] = pd.to_numeric(X_seq_df[col], errors="coerce").fillna(fill_value)
        X_seq = X_seq_df.values.reshape(len(df), 5, len(sequence_features) // 5)

        flat_numerical_features = [
            "age",
            "weight_carry",
            "horse_num",
            "horse_weight",
            "weight_change",
        ]
        flat_categorical_features = ["sex", "jockey", "field", "weather", "place"]
        X_flat_num = df[flat_numerical_features].copy()
        for col in X_flat_num.columns:
            # Use imputation_values if available, otherwise fallback to 0
            fill_value = imputation_values.get(col, 0) if imputation_values else 0
            X_flat_num[col] = pd.to_numeric(X_flat_num[col], errors="coerce").fillna(fill_value)
        X_flat_cat = df[flat_categorical_features].copy()
        for col in X_flat_cat.columns:
            # Use imputation_values if available, otherwise fallback to "missing"
            fill_value = imputation_values.get(col, "missing") if imputation_values else "missing"
            X_flat_cat[col] = X_flat_cat[col].astype(str).fillna(fill_value)
        X_flat_cat_dummies = pd.get_dummies(
            X_flat_cat, columns=flat_categorical_features
        )

        X_flat = pd.concat([X_flat_num, X_flat_cat_dummies], axis=1)

        if flat_features_columns:
            X_flat = X_flat.reindex(columns=flat_features_columns, fill_value=0)
        return [X_seq, X_flat]
    else:
        raise ValueError("This function is designed for CNN prediction.")


def load_model(target_mode="default"):
    model_path = get_model_path("cnn", target_mode)
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        
        flat_features_json_path = model_path + ".flat_features.json"
        flat_features_columns = (
            json.load(open(flat_features_json_path, "r")) if os.path.exists(flat_features_json_path) else None
        )
        
        imputation_values_json_path = model_path + ".imputation_values.json"
        imputation_values = (
            json.load(open(imputation_values_json_path, "r")) if os.path.exists(imputation_values_json_path) else None
        )
        
        return model, flat_features_columns, imputation_values
    else:
        print(f"CNN Model not found at {model_path}. Please train the model first.")
        return None, None, None


def predict_rank(model, new_data_processed, target_mode="default"):
    if not model or new_data_processed[0].size == 0:
        return None
    
    probabilities = model.predict(new_data_processed)
    
    if target_mode == "default":
        # Class indices: 0 for 1st, 1 for 2-3rd, 2 for Others
        
        # Create a list of (horse_index, prob_1st, prob_2_3rd)
        horse_probs = []
        for i in range(len(probabilities)):
            horse_probs.append((i, probabilities[i][0], probabilities[i][1])) # (index, prob_1st, prob_2-3rd)
        
        # Sort by prob_1st in descending order
        horse_probs_sorted_by_1st = sorted(horse_probs, key=lambda x: x[1], reverse=True)
        
        # Initialize all predictions to Others
        final_predictions = ["Others"] * len(probabilities)
        
        # Assign 1st place
        if len(horse_probs_sorted_by_1st) > 0:
            first_place_horse_idx = horse_probs_sorted_by_1st[0][0]
            final_predictions[first_place_horse_idx] = "1st"
            
            # Remove the 1st place horse from consideration for 2-3rd
            remaining_horses = [hp for hp in horse_probs_sorted_by_1st if hp[0] != first_place_horse_idx]
            
            # Sort remaining by prob_2-3rd in descending order
            remaining_horses_sorted_by_2_3rd = sorted(remaining_horses, key=lambda x: x[2], reverse=True)
            
            # Assign 2-3rd places (up to 2 horses)
            assigned_2_3rd_count = 0
            for hp in remaining_horses_sorted_by_2_3rd:
                if assigned_2_3rd_count < 2:
                    final_predictions[hp[0]] = "2-3rd"
                    assigned_2_3rd_count += 1
                else:
                    break
        
        return final_predictions
        
    elif target_mode == "top3":
        # Class indices: 0 for 1-3rd, 1 for Others
        # Similar logic, but assign top 3 horses to "1-3rd"
        horse_probs = []
        for i in range(len(probabilities)):
            horse_probs.append((i, probabilities[i][0])) # (index, prob_1-3rd)
        
        horse_probs_sorted_by_1_3rd = sorted(horse_probs, key=lambda x: x[1], reverse=True)
        
        final_predictions = ["Others"] * len(probabilities)
        
        assigned_1_3rd_count = 0
        for hp in horse_probs_sorted_by_1_3rd:
            if assigned_1_3rd_count < 3:
                final_predictions[hp[0]] = "1-3rd"
                assigned_1_3rd_count += 1
            else:
                break
        return final_predictions
    else:
        raise ValueError(f"Unknown target_mode: {target_mode}")


# --- Main Logic for Prediction ---
def predict_race_from_url(race_url: str, target_mode="default"):
    print(
        f"Starting scraping and prediction for race: {race_url} with target mode: {target_mode}..."
    )
    model, flat_features_columns, imputation_values = load_model(target_mode=target_mode)
    if not model:
        return

    soup_overview = fetch_page(race_url)
    if not soup_overview:
        return

    try:
        title_text = soup_overview.find("title").text
        date_match = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", title_text)
        year, month, day_of_month = [int(g) for g in date_match.groups()]
        target_race_date = pd.to_datetime(f"{year}-{month}-{day_of_month}")

        race_num_element = soup_overview.find(class_="RaceList_Item01").find(
            class_="RaceNum"
        )
        race_num = int(re.search(r"(\d+)R", race_num_element.text).group(1))

        race_data01_text = soup_overview.find(class_="RaceData01").text
        race_data02_spans = [
            s.text for s in soup_overview.find(class_="RaceData02").find_all("span")
        ]

        field_dist_match = re.search(r"(芝|ダ|障)\s*(\d+)m", race_data01_text)
        field_type, distance = field_dist_match.groups()
        weather_match = re.search(r"天候:(\S)", race_data01_text)
        weather = weather_match.group(1) if weather_match else ""
        field_cond_match = re.search(r"馬場:(\S)", race_data01_text)
        field_cond = field_cond_match.group(1) if field_cond_match else ""

        kai = int(re.sub(r"\D", "", race_data02_spans[0]))
        place_name = race_data02_spans[1]
        day_of_kai = int(re.sub(r"\D", "", race_data02_spans[2]))

    except Exception as e:
        print(f"Could not parse race overview info: {e}. Exiting.")
        return

    base_race_info = {
        "year": year,
        "date": f"{month}/{day_of_month}",
        "month": month,
        "race_num": race_num,
        "field": field_type,
        "dist": int(distance),
        "turn": get_race_turn(place_name),
        "weather": weather,
        "field_cond": field_cond,
        "kai": kai,
        "day": day_of_kai,
        "place": place_name,
    }

    df_detail = parse_race_details(soup_overview)
    if df_detail.empty:
        return

    base_race_info["sum_num"] = len(df_detail)
    all_race_data = []

    # parquetファイルを読み込む
    try:
        df_parquet = pd.read_parquet(DATA_FILE)
        # 'date'カラムをdatetime型に変換
        df_parquet["date"] = df_parquet.apply(
            lambda row: pd.to_datetime(
                f"{int(row['year'])}/{row['date'].replace(' ', '')}",
                format="%Y/%m/%d",
                errors="coerce",
            ),
            axis=1,
        )
        df_parquet.dropna(subset=["date"], inplace=True)  # 無効な日付を削除

    except Exception as e:
        print(f"Error loading or processing parquet file: {e}. Exiting.")
        return

    for _, detail_row in df_detail.iterrows():
        horse_name = detail_row.get("馬名", "")

        # parquetから馬の最新データを取得
        # 推論対象レース日付以前で、その馬の最も新しいレースのデータを探す
        horse_data_from_parquet = (
            df_parquet[
                (df_parquet["horse_name"] == horse_name)
                & (df_parquet["date"] < target_race_date)
            ]
            .sort_values(by="date", ascending=False)
            .head(1)
        )

        if horse_data_from_parquet.empty:
            print(
                f"Warning: No historical data found for horse '{horse_name}' in parquet. Filling with NaNs."
            )
            extracted_index_data = {col: np.nan for col in INDEX_HEAD}
            parsed_previous_race_data = {col: np.nan for col in PREV_RACE_HEAD}
        else:
            latest_horse_row = horse_data_from_parquet.iloc[0]
            extracted_index_data = {
                col: latest_horse_row.get(col, np.nan) for col in INDEX_HEAD
            }
            parsed_previous_race_data = {
                col: latest_horse_row.get(col, np.nan) for col in PREV_RACE_HEAD
            }

        horse_weight, weight_change = process_horse_weight(
            detail_row.get("horse_weight_change", "---")
        )
        sex, age = re.match(r"(.)(\d+)", detail_row["sex_age"]).groups()

        full_race_info = {
            **base_race_info,
            **{
                "prize": 0,
                "rank": -1,
                "time": "",
                "l_days": "",
                "horse_num": int(detail_row.get("horse_num", -1)),
                "horse_name": horse_name,
                "sex": sex,
                "age": int(age),
                "weight_carry": float(detail_row.get("weight_carry", 0)),
                "horse_weight": horse_weight,
                "weight_change": weight_change,
                "jockey": detail_row.get("jockey", ""),
            },
        }
        full_race_info.update(extracted_index_data)
        full_race_info.update(parsed_previous_race_data)

        all_race_data.append(full_race_info)

    if not all_race_data:
        return
    df_final_for_prediction = pd.DataFrame(all_race_data, columns=FINAL_COLS)

    print("Preprocessing data for CNN prediction...")
    X_predict = preprocess_data_for_prediction(
        df_final_for_prediction.copy(),
        model_type="cnn",
        flat_features_columns=flat_features_columns,
        imputation_values=imputation_values
    )

    if X_predict[0].size == 0:
        return
    predicted_ranks = predict_rank(model, X_predict, target_mode=target_mode)

    print("\n--- Prediction Results ---")
    for i, (_, row) in enumerate(df_final_for_prediction.iterrows()):
        print(f"馬名: {row['horse_name']}, 予測順位カテゴリ: {predicted_ranks[i]}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        race_url_to_predict = sys.argv[1]
        target_mode_to_use = "default"
        if len(sys.argv) > 2 and sys.argv[2] == "top3":
            target_mode_to_use = "top3"
        predict_race_from_url(race_url_to_predict, target_mode=target_mode_to_use)
    else:
        print("Usage: python -m scripts.predict_latest_race <race_url> [target_mode]")
        print(
            "Example: python -m scripts.predict_latest_race https://race.netkeiba.com/race/shutuba.html?race_id=202405040811"
        )
        print("target_mode can be 'default' or 'top3'. Default is 'default'.")
