import sys
import pandas as pd
import re
import numpy as np

# GPUが利用可能か確認し、設定を行う
import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU is available and memory growth is enabled.")
    except:
        print("Could not set GPU memory growth.")
else:
    print("No GPU devices found. Using CPU.")

from scripts.prediction_utils.model_loader import load_all_models
from scripts.prediction_utils.constants import (
    URL_PREFIX,
    DATA_FILE,
    INDEX_HEAD,
    PREV_RACE_HEAD,
    FINAL_COLS,
)
from scripts.prediction_utils.scraper import fetch_race_data_with_playwright
from scripts.prediction_utils.data_preprocessor import (
    preprocess_data_for_prediction,
    get_race_turn,
    process_horse_weight,
)
from scripts.prediction_utils.predictor import predict_with_all_models
from scripts.prediction_utils.ensembler import ensemble_predictions
from scripts.prediction_utils.value_betting import identify_value_bets


# --- Main Logic for Ensemble Prediction ---
def predict_ensemble_race_from_url(race_url: str, target_mode="default"):
    print(
        f"Starting scraping and ensemble prediction for race: {race_url} with target mode: {target_mode}..."
    )

    all_models = load_all_models()
    if not all_models or not all_models[target_mode]:
        print(f"No models loaded for target mode: {target_mode}. Exiting.")
        return

    df_detail, soup_overview = fetch_race_data_with_playwright(race_url)
    if df_detail.empty or not soup_overview:
        print("Failed to fetch or parse race data with Playwright. Exiting.")
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
        weather = weather_match.group(1) if weather_match else "missing"
        field_cond_match = re.search(r"馬場:(\S)", race_data01_text)
        field_cond = field_cond_match.group(1) if field_cond_match else "missing"

        kai = int(re.search(r"\d+", race_data02_spans[0]).group(0))
        place_name = race_data02_spans[1]
        day_of_kai = int(re.search(r"\d+", race_data02_spans[2]).group(0))

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
        horse_name = detail_row.get("horse_name", "")

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
                f"Warning: No historical data found for horse '{horse_name}' in parquet. Filling with defaults."
            )
            # Define default values based on expected data types
            extracted_index_data = {col: np.nan for col in INDEX_HEAD}

            parsed_previous_race_data = {}
            object_cols_in_prev = ["place", "weather", "field", "condi"]
            for col in PREV_RACE_HEAD:
                # Check if any of the object column substrings are in the column name
                if any(sub in col for sub in object_cols_in_prev):
                    parsed_previous_race_data[col] = (
                        "missing"  # Default for object/string types
                    )
                else:
                    parsed_previous_race_data[col] = np.nan  # Default for numeric types
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
                "odds": detail_row.get("odds", 0.0), # オッズ情報を追加
            },
        }
        full_race_info.update(extracted_index_data)
        full_race_info.update(parsed_previous_race_data)

        all_race_data.append(full_race_info)

    if not all_race_data:
        return
    df_final_for_prediction = pd.DataFrame(all_race_data, columns=FINAL_COLS)

    print("Preprocessing data for all models...")
    preprocessed_data_for_models = {}

    # Preprocess for RF models
    for horse_info in ["included", "excluded"]:
        model_key = f"rf_{horse_info}"
        if model_key in all_models[target_mode]:
            rf_target_maps = all_models[target_mode][model_key].get("target_maps")
            # Exclude horse info if needed for preprocessing
            df_rf_prep = df_final_for_prediction.copy()
            if horse_info == "excluded":
                df_rf_prep = df_rf_prep.drop(
                    columns=["horse_num", "horse_weight", "weight_change"],
                    errors="ignore",
                )
            preprocessed_data_for_models[model_key] = preprocess_data_for_prediction(
                df_rf_prep,
                "rf",
                target_maps=rf_target_maps,
                expected_columns=all_models[target_mode][model_key].get(
                    "expected_columns"
                ),
            )

    # Preprocess for LGBM models
    for horse_info in ["included", "excluded"]:
        model_key = f"lgbm_{horse_info}"
        if model_key in all_models[target_mode]:
            df_lgbm_prep = df_final_for_prediction.copy()
            if horse_info == "excluded":
                df_lgbm_prep = df_lgbm_prep.drop(
                    columns=["horse_num", "horse_weight", "weight_change"],
                    errors="ignore",
                )
            preprocessed_data_for_models[model_key] = preprocess_data_for_prediction(
                df_lgbm_prep,
                "lgbm",
                expected_columns=all_models[target_mode][model_key].get(
                    "expected_columns"
                ),
                categorical_features_with_categories=all_models[target_mode][
                    model_key
                ].get("categorical_features_with_categories"),
            )

    # Preprocess for CNN model
    model_key = "cnn_included"
    if model_key in all_models[target_mode]:
        cnn_flat_features = all_models[target_mode][model_key].get("flat_features")
        cnn_imputation_values = all_models[target_mode][model_key].get(
            "imputation_values"
        )
        preprocessed_data_for_models[model_key] = preprocess_data_for_prediction(
            df_final_for_prediction.copy(),
            "cnn",
            flat_features_columns=cnn_flat_features,
            imputation_values=cnn_imputation_values,
        )

    print("Making predictions with individual models...")

    # --- DEBUGGING STEP ---
    print("\n--- DEBUGGING CATEGORICAL FEATURES ---")
    for model_key, data in preprocessed_data_for_models.items():
        if 'lgbm' in model_key:
            print(f"\n[DEBUG] Inspecting data for: {model_key}")
            if isinstance(data, pd.DataFrame):
                cat_cols = [col for col in data.columns if data[col].dtype.name == 'category']
                if not cat_cols:
                    print("No categorical columns found in preprocessed data.")
                for col in cat_cols:
                    print(f"  - Column: {col}")
                    print(f"    - Dtype: {data[col].dtype}")
                    # Limit printing categories if there are too many
                    if len(data[col].cat.categories) > 50:
                         print(f"    - Categories ({len(data[col].cat.categories)} total): {data[col].cat.categories[:25].tolist()}... (truncated)")
                    else:
                         print(f"    - Categories: {data[col].cat.categories.tolist()}")
            else:
                print("Preprocessed data is not a DataFrame.")
    print("--- END DEBUGGING ---\n")
    # --- END DEBUGGING STEP ---

    individual_predictions = predict_with_all_models(
        all_models, preprocessed_data_for_models, target_mode
    )

    print("Ensembling predictions...")
    final_predicted_ranks = ensemble_predictions(individual_predictions, target_mode)

    print("\n--- Ensemble Prediction Results ---")
    for i, (_, row) in enumerate(df_final_for_prediction.iterrows()):
        print(
            f"馬名: {row['horse_name']}, 予測順位カテゴリ: {final_predicted_ranks[i]}"
        )

    # Identify and display value bets
    # Check if odds data is available and valid before proceeding with value betting analysis
    if (
        "odds" in df_final_for_prediction.columns
        and not df_final_for_prediction["odds"].astype(str).str.contains("---").all()
        and not df_final_for_prediction["odds"].dropna().empty
    ):
        value_bets_df = identify_value_bets(
            df_final_for_prediction, individual_predictions[all_models[target_mode]["rf_included"]["model_path"]], target_mode
        )

        print("\n--- Value Betting Analysis ---")
        if not value_bets_df.empty:
            for _, vb_row in value_bets_df.iterrows():
                print(
                    f"馬名: {vb_row['horse_name']}, オッズ: {vb_row['odds']:.2f}, 予測確率: {vb_row['model_prob']:.2%}, 期待値: {vb_row['expected_value']:.2f}, バリューベット: {vb_row['is_value_bet']}"
                )
        else:
            print("No value bets identified or odds data missing.")
    else:
        print("\n--- Value Betting Analysis ---")
        print(
            "Odds data not available or invalid for this race. Skipping value betting analysis."
        )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        race_url_to_predict = sys.argv[1]
        target_mode_to_use = "default"
        if len(sys.argv) > 2 and sys.argv[2] == "top3":
            target_mode_to_use = "top3"
        predict_ensemble_race_from_url(
            race_url_to_predict, target_mode=target_mode_to_use
        )
    else:
        print("Usage: python -m scripts.predict_ensemble <race_url> [target_mode]")
        print(
            "Example: python -m scripts.predict_ensemble https://race.netkeiba.com/race/shutuba.html?race_id=202405040811"
        )
        print("target_mode can be 'default' or 'top3'. Default is 'default'.")
