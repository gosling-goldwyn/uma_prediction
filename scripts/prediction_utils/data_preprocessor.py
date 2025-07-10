import pandas as pd
import numpy as np
import re

from scripts.prediction_utils.constants import COUNTERCLOCKWISE_PLACES, CLOCKWISE_PLACES

def get_race_turn(place_name):
    if place_name in COUNTERCLOCKWISE_PLACES:
        return "左"
    elif place_name in CLOCKWISE_PLACES:
        return "右"
    else:
        return "直線"

def process_horse_weight(horse_weight_change_str):
    if horse_weight_change_str == "---":
        return np.nan, np.nan
    match = re.match(r"(\d+)\\(([-+]?\d+)\\)", horse_weight_change_str)
    if match:
        horse_weight = int(match.group(1))
        weight_change = int(match.group(2))
        return horse_weight, weight_change
    return np.nan, np.nan

def preprocess_data_for_prediction(df, model_type, target_maps=None, flat_features_columns=None, imputation_values=None, expected_columns=None, categorical_features_with_categories=None):
    df_processed = df.copy()

    # Common preprocessing steps (from train.py's preprocess_data)
    # Fill missing values for all types
    for col in df_processed.columns:
        if pd.api.types.is_numeric_dtype(df_processed[col]):
            if df_processed[col].isnull().all():
                df_processed[col] = df_processed[col].fillna(0)
            else:
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())
        else:
            if df_processed[col].isnull().all():
                df_processed[col] = df_processed[col].fillna("missing")
            else:
                df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])

    if model_type == "rf":
        high_cardinality_features = ["horse_name", "jockey"]
        for col in high_cardinality_features:
            if col in df_processed.columns and target_maps and col in target_maps:
                df_processed[col] = df_processed[col].map(target_maps[col]).fillna(0) # Fallback to 0 if all NaN
            elif col in df_processed.columns: # If no target_maps, fill with 0 or a default
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0)

        low_cardinality_features = [
            col
            for col in df_processed.columns
            if df_processed[col].dtype == "object" and col not in high_cardinality_features
        ]
        df_processed = pd.get_dummies(df_processed, columns=low_cardinality_features, dummy_na=False)
        if expected_columns is not None:
            df_processed = df_processed.reindex(columns=expected_columns, fill_value=0)
        return df_processed

    elif model_type == "lgbm":
        categorical_features = [col for col in df_processed.columns if df_processed[col].dtype == "object"]
        for col in categorical_features:
            # NaNを'missing'として扱う
            df_processed[col] = df_processed[col].fillna('missing')
            if categorical_features_with_categories and col in categorical_features_with_categories:
                # 訓練時に使用したカテゴリを明示的に指定
                df_processed[col] = pd.Categorical(df_processed[col], categories=categorical_features_with_categories[col])
            else:
                df_processed[col] = df_processed[col].astype("category")
        if expected_columns is not None:
            df_processed = df_processed.reindex(columns=expected_columns, fill_value=0)
        return df_processed

    elif model_type == "cnn":
        sequence_features = [
            "1st_speed_idx", "2nd_speed_idx", "3rd_speed_idx", "4th_speed_idx", "5th_speed_idx",
            "1st_lead_idx", "2nd_lead_idx", "3rd_lead_idx", "4th_lead_idx", "5th_lead_idx",
            "1st_pace_idx", "2nd_pace_idx", "3rd_pace_idx", "4th_pace_idx", "5th_pace_idx",
            "1st_rising_idx", "2nd_rising_idx", "3rd_rising_idx", "4th_rising_idx", "5th_rising_idx",
        ]
        X_seq_df = df_processed[sequence_features].copy()
        for col in sequence_features:
            fill_value = imputation_values.get(col, 0) if imputation_values else 0
            X_seq_df[col] = pd.to_numeric(X_seq_df[col], errors="coerce").fillna(fill_value)
        X_seq = X_seq_df.values.reshape(len(df_processed), 5, len(sequence_features) // 5)

        flat_numerical_features = [
            "age", "weight_carry", "horse_num", "horse_weight", "weight_change",
        ]
        flat_categorical_features = ["sex", "jockey", "field", "weather", "place"]

        X_flat_num = df_processed[flat_numerical_features].copy()
        for col in X_flat_num.columns:
            fill_value = imputation_values.get(col, 0) if imputation_values else 0
            X_flat_num[col] = pd.to_numeric(X_flat_num[col], errors="coerce").fillna(fill_value)

        X_flat_cat = df_processed[flat_categorical_features].copy()
        for col in X_flat_cat.columns:
            fill_value = imputation_values.get(col, "missing") if imputation_values else "missing"
            X_flat_cat[col] = X_flat_cat[col].astype(str).fillna(fill_value)
        X_flat_cat_dummies = pd.get_dummies(X_flat_cat, columns=flat_categorical_features)

        X_flat = pd.concat([X_flat_num, X_flat_cat_dummies], axis=1)

        if flat_features_columns:
            X_flat = X_flat.reindex(columns=flat_features_columns, fill_value=0)
        return [X_seq, X_flat]
    else:
        raise ValueError("Invalid model_type specified.")