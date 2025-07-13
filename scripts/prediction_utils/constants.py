import numpy as np

# --- Constants ---
URL_PREFIX = "https://db.netkeiba.com"
DATA_FILE = "data/index_dataset.parquet"

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
        "odds",
    ]
    + INDEX_HEAD
    + PREV_RACE_HEAD
)
