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

def preprocess_data_for_prediction(df, model_type, target_maps=None, flat_features_columns=None, imputation_values=None, expected_columns=None, categorical_features_with_categories=None, imputation_map=None):
    df_processed = df.copy()

    # Drop columns that are not features for prediction
    # Added 'odds' to the list of columns to be dropped.
    cols_to_drop = ["rank", "time", "prize", "date", "odds"]
    df_processed = df_processed.drop(columns=[col for col in cols_to_drop if col in df_processed.columns], errors='ignore')

    # Convert 'l_days' to numeric, coercing errors to NaN
    if 'l_days' in df_processed.columns:
        df_processed['l_days'] = pd.to_numeric(df_processed['l_days'], errors='coerce')

    # Feature Engineering
    rank_cols = ['1st_rank', '2nd_rank', '3rd_rank', '4th_rank', '5th_rank']
    speed_idx_cols = ['1st_speed_idx', '2nd_speed_idx', '3rd_speed_idx', '4th_speed_idx', '5th_speed_idx']

    # Time-series features
    df_processed['avg_rank_past_5'] = df_processed[rank_cols].mean(axis=1)
    df_processed['win_rate_past_5'] = (df_processed[rank_cols] == 1).sum(axis=1) / df_processed[rank_cols].notna().sum(axis=1)
    df_processed['top_3_rate_past_5'] = (df_processed[rank_cols] <= 3).sum(axis=1) / df_processed[rank_cols].notna().sum(axis=1)
    df_processed['avg_speed_idx_past_5'] = df_processed[speed_idx_cols].mean(axis=1)
    df_processed['max_speed_idx_past_5'] = df_processed[speed_idx_cols].max(axis=1)
    df_processed['std_rank_past_5'] = df_processed[rank_cols].std(axis=1)

    # Interaction features
    df_processed['jockey_place'] = df_processed['jockey'].astype(str) + '_' + df_processed['place'].astype(str)
    df_processed['horse_place'] = df_processed['horse_name'].astype(str) + '_' + df_processed['place'].astype(str)
    df_processed['horse_dist'] = df_processed['horse_name'].astype(str) + '_' + df_processed['dist'].astype(str)
    df_processed['place_dist'] = df_processed['place'].astype(str) + '_' + df_processed['dist'].astype(str)

    # Domain knowledge features
    if 'horse_weight' in df_processed.columns and 'weight_carry' in df_processed.columns:
        df_processed['weight_ratio'] = df_processed['weight_carry'] / df_processed['horse_weight']
    elif 'weight_ratio' not in df_processed.columns:
        df_processed['weight_ratio'] = 0

    # Defragment the DataFrame to avoid PerformanceWarning
    df_processed = df_processed.copy()

    # Fill missing values using the imputation_map from training
    if imputation_map:
        for col, value in imputation_map.items():
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].fillna(value)

    if model_type == "rf":
        high_cardinality_features = [
            "horse_name",
            "jockey",
            "jockey_place",
            "horse_place",
            "horse_dist",
        ]
        for col in high_cardinality_features:
            if col in df_processed.columns and target_maps and col in target_maps:
                # Use the mean of the target map as a fallback for unseen values
                fallback_value = np.mean(list(target_maps[col].values()))
                df_processed[col] = df_processed[col].map(target_maps[col]).fillna(fallback_value)
            elif col in df_processed.columns: # If no target_maps, fill with 0 or a default
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0)

        low_cardinality_features = [
            col
            for col in df_processed.columns
            if df_processed[col].dtype == "object" and col not in high_cardinality_features
        ]
        df_processed = pd.get_dummies(df_processed, columns=low_cardinality_features, dummy_na=False)

        # Align columns with the training set
        if expected_columns is not None:
            # Reindex to match the training columns, filling missing ones with 0
            df_processed = df_processed.reindex(columns=expected_columns, fill_value=0)

        return df_processed

    elif model_type == "lgbm":
        # Ensure all expected categorical features are present, even if all values are missing
        if categorical_features_with_categories:
            for col in categorical_features_with_categories.keys():
                if col not in df_processed.columns:
                    df_processed[col] = 'missing' # Add missing column with default value

        # Identify all object columns in the current DataFrame
        current_object_cols = [col for col in df_processed.columns if df_processed[col].dtype == "object"]

        for col in current_object_cols:
            # Convert empty strings to NaN first to be handled by fillna
            df_processed[col] = df_processed[col].replace('', np.nan)
            # Fill any existing NaNs with 'missing'
            df_processed[col] = df_processed[col].fillna('missing')

            if categorical_features_with_categories and col in categorical_features_with_categories:
                known_categories = categorical_features_with_categories[col]
                
                # Set the column to the CategoricalDtype with categories from training.
                # This ensures that the categories match exactly what LGBM expects.
                # Any value in the data that is not in `known_categories` will be converted to NaN.
                df_processed[col] = pd.Categorical(df_processed[col], categories=known_categories)
                
                # After setting the categories, there might be NaNs for values that were not in the training categories.
                # We fill these NaNs with our designated 'missing' category value.
                if df_processed[col].isnull().any():
                    # The 'missing' category might not be present if all values were known.
                    # We ensure it exists before trying to fill with it.
                    if 'missing' not in df_processed[col].cat.categories:
                         df_processed[col] = df_processed[col].cat.add_categories('missing')
                    df_processed[col] = df_processed[col].fillna('missing')
            else:
                # Fallback for columns not in the saved categories list
                df_processed[col] = df_processed[col].astype('category')

        # Reindex to ensure column order and presence matches training data
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
            # Use imputation_values for sequence features
            fill_value = imputation_values.get(col, 0) if imputation_values and col in imputation_values else 0
            X_seq_df[col] = pd.to_numeric(X_seq_df[col], errors="coerce").fillna(fill_value)
        X_seq = X_seq_df.values.reshape(len(df_processed), 5, len(sequence_features) // 5)

        flat_numerical_features = [
            "age", "weight_carry", "horse_num", "horse_weight", "weight_change",
        ]
        flat_categorical_features = ["sex", "jockey", "field", "weather", "place"]

        X_flat_num = df_processed[flat_numerical_features].copy()
        for col in X_flat_num.columns:
            # Use imputation_values for flat numerical features
            fill_value = imputation_values.get(col, 0) if imputation_values and col in imputation_values else 0
            X_flat_num[col] = pd.to_numeric(X_flat_num[col], errors="coerce").fillna(fill_value)

        X_flat_cat = df_processed[flat_categorical_features].copy()
        for col in X_flat_cat.columns:
            # Use imputation_values for flat categorical features
            fill_value = imputation_values.get(col, "missing") if imputation_values and col in imputation_values else "missing"
            X_flat_cat[col] = X_flat_cat[col].astype(str).fillna(fill_value)
        X_flat_cat_dummies = pd.get_dummies(X_flat_cat, columns=flat_categorical_features)

        X_flat = pd.concat([X_flat_num, X_flat_cat_dummies], axis=1)

        if flat_features_columns:
            X_flat = X_flat.reindex(columns=flat_features_columns, fill_value=0)
        return [X_seq, X_flat]
    else:
        raise ValueError("Invalid model_type specified.")