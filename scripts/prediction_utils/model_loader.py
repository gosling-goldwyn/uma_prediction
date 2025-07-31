import os
import json
import joblib
import tensorflow as tf

import os
import json
import joblib
import tensorflow as tf
import re

from scripts.model_training.training_utils import FocalLoss # FocalLossをインポート

# --- Helper function to parse model filenames ---
def _parse_model_filename(filename):
    # Example: rf_uma_prediction_model_default_place_東京.pkl
    # Example: cnn_uma_prediction_model_default_with_cat_place_東京.keras
    # Example: lgbm_uma_prediction_model_default_no_horse_info.txt

    match = re.match(r"^(rf|lgbm|cnn)_uma_prediction_model_([a-zA-Z0-9]+)(_no_horse_info)?(_with_cat)?(_place_[^.]+)?\.(pkl|txt|keras)$", filename)
    if not match:
        return None

    model_type = match.group(1)
    target_mode = match.group(2)
    horse_info = "excluded" if match.group(3) else "included"
    fine_tune_suffix_part = match.group(5)
    extension = match.group(6)

    model_key_parts = [model_type]
    if model_type != "cnn": # RF and LGBM can have horse_info excluded
        model_key_parts.append(horse_info)

    if fine_tune_suffix_part:
        # Remove leading underscore
        fine_tune_suffix = fine_tune_suffix_part[1:]
        model_key_parts.append(fine_tune_suffix)

    # Construct a unique key for the models dictionary (e.g., rf_included_place_東京)
    model_dict_key = "_".join(model_key_parts)

    return {
        "model_type": model_type,
        "target_mode": target_mode,
        "horse_info": horse_info,
        "fine_tune_suffix": fine_tune_suffix_part if fine_tune_suffix_part else "",
        "extension": extension,
        "model_dict_key": model_dict_key
    }


def load_all_models():
    models = {"default": {}, "top3": {}}
    model_dir = "models"

    if not os.path.exists(model_dir):
        print(f"Model directory not found: {model_dir}")
        return models

    for filename in os.listdir(model_dir):
        parsed_info = _parse_model_filename(filename)
        if not parsed_info:
            continue

        model_type = parsed_info["model_type"]
        target_mode = parsed_info["target_mode"]
        horse_info = parsed_info["horse_info"]
        fine_tune_suffix = parsed_info["fine_tune_suffix"]
        extension = parsed_info["extension"]
        model_dict_key = parsed_info["model_dict_key"]
        
        model_path = os.path.join(model_dir, filename)

        try:
            if model_type == "rf":
                if os.path.exists(model_path):
                    model_data = joblib.load(model_path)
                    model = model_data['model']
                    calibrators = model_data.get('calibrators') # Use .get for safe access
                    feature_columns_path = model_path + ".feature_columns.json"
                    target_maps_path = model_path + ".target_maps.json"
                    imputation_path = model_path + ".imputation.json"

                    feature_columns = None
                    if os.path.exists(feature_columns_path):
                        with open(feature_columns_path, 'r') as f:
                            feature_columns = json.load(f)

                    target_maps = None
                    if os.path.exists(target_maps_path):
                        with open(target_maps_path, 'r') as f:
                            target_maps = json.load(f)

                    imputation_map = None
                    if os.path.exists(imputation_path):
                        with open(imputation_path, 'r') as f:
                            imputation_map = json.load(f)

                    models[target_mode][model_dict_key] = {
                        "model": model,
                        "calibrators": calibrators,
                        "feature_columns": feature_columns,
                        "target_maps": target_maps,
                        "imputation_map": imputation_map,
                        "model_path": model_path,
                        "model_type": model_type
                    }
                    print(f"Loaded RF model: {os.path.basename(model_path)}")

            elif model_type == "lgbm":
                if os.path.exists(model_path):
                    lgbm_model = joblib.load(model_path)
                    lgbm_feature_columns = None
                    lgbm_feature_columns_path = model_path.replace(f".{extension}", ".feature_columns.json")
                    if os.path.exists(lgbm_feature_columns_path):
                        with open(lgbm_feature_columns_path, "r") as f:
                            lgbm_feature_columns = json.load(f)
                    from scripts.data_preprocessing.lgbm_categorical_processor import load_lgbm_categorical_features
                    lgbm_categorical_features_with_categories = load_lgbm_categorical_features(model_path)
                    if lgbm_categorical_features_with_categories is None:
                        print(f"Warning: No categorical features info found for LGBM model: {model_path}. This might cause issues.")

                    imputation_map = None
                    imputation_path = model_path.replace(f".{extension}", ".imputation.json")
                    if os.path.exists(imputation_path):
                        with open(imputation_path, 'r') as f:
                            imputation_map = json.load(f)

                    models[target_mode][model_dict_key] = {
                        "model": lgbm_model,
                        "expected_columns": lgbm_feature_columns,
                        "categorical_features_with_categories": lgbm_categorical_features_with_categories,
                        "imputation_map": imputation_map,
                        "model_path": model_path,
                        "model_type": model_type
                    }
                    print(f"Loaded LGBM model: {model_path}")

            elif model_type == "cnn":
                if os.path.exists(model_path):
                    cnn_model = tf.keras.models.load_model(model_path, custom_objects={'FocalLoss': FocalLoss})
                    cnn_flat_features_path = model_path.replace(f".{extension}", ".flat_features.json")
                    cnn_imputation_values_path = model_path.replace(f".{extension}", ".imputation_values.json")
                    
                    cnn_flat_features = None
                    if os.path.exists(cnn_flat_features_path):
                        with open(cnn_flat_features_path, "r") as f:
                            cnn_flat_features = json.load(f)
                    
                    cnn_imputation_values = None
                    if os.path.exists(cnn_imputation_values_path):
                        with open(cnn_imputation_values_path, "r") as f:
                            cnn_imputation_values = json.load(f)

                    imputation_map = None
                    imputation_path = model_path.replace(f".{extension}", ".imputation.json")
                    if os.path.exists(imputation_path):
                        with open(imputation_path, 'r') as f:
                            imputation_map = json.load(f)

                    models[target_mode][model_dict_key] = {
                        "model": cnn_model,
                        "flat_features": cnn_flat_features,
                        "imputation_values": cnn_imputation_values,
                        "imputation_map": imputation_map,
                        "model_path": model_path,
                        "model_type": model_type
                    }
                    print(f"Loaded CNN model: {model_path}")

        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
            
    return models
