import os
import json
import joblib
import tensorflow as tf

from scripts.model_training.train import get_model_path
from scripts.data_preprocessing.lgbm_categorical_processor import load_lgbm_categorical_features
from scripts.model_training.training_utils import FocalLoss # FocalLossをインポート

def load_all_models():
    models = {}
    target_modes = ["default", "top3"]
    horse_info_options = ["included", "excluded"] # Only for rf and lgbm

    for target_mode in target_modes:
        models[target_mode] = {}

        # Load RF and LGBM models (with and without horse info)
        for horse_info in horse_info_options:
            # RF
            rf_model_path = get_model_path("rf", target_mode, horse_info)
            if os.path.exists(rf_model_path):
                try:
                    loaded_obj = joblib.load(rf_model_path)
                    rf_model = loaded_obj["model"]
                    rf_calibrators = loaded_obj["calibrators"]

                    rf_target_maps_path = rf_model_path + ".target_maps.json"
                    rf_target_maps = None
                    if os.path.exists(rf_target_maps_path):
                        with open(rf_target_maps_path, "r") as f:
                            rf_target_maps = json.load(f)
                    rf_feature_columns = None
                    rf_feature_columns_path = rf_model_path + ".feature_columns.json"
                    if os.path.exists(rf_feature_columns_path):
                        with open(rf_feature_columns_path, "r") as f:
                            rf_feature_columns = json.load(f)
                    models[target_mode][f"rf_{horse_info}"] = {
                        "model": rf_model,
                        "calibrators": rf_calibrators,
                        "target_maps": rf_target_maps,
                        "expected_columns": rf_feature_columns,
                        "model_path": rf_model_path # Add model_path
                    }
                    print(f"Loaded RF model: {rf_model_path}")
                except Exception as e:
                    print(f"Error loading RF model {rf_model_path}: {e}")
            else:
                print(f"RF model not found: {rf_model_path}")

            # LGBM
            lgbm_model_path = get_model_path("lgbm", target_mode, horse_info)
            if os.path.exists(lgbm_model_path):
                try:
                    lgbm_model = joblib.load(lgbm_model_path)
                    # LGBM categorical features are handled during training and saved within the model
                    lgbm_feature_columns = None
                    lgbm_feature_columns_path = lgbm_model_path + ".feature_columns.json"
                    if os.path.exists(lgbm_feature_columns_path):
                        with open(lgbm_feature_columns_path, "r") as f:
                            lgbm_feature_columns = json.load(f)
                    # カテゴリカル特徴量のカテゴリ情報をロード
                    lgbm_categorical_features_with_categories = load_lgbm_categorical_features(lgbm_model_path)
                    if lgbm_categorical_features_with_categories is None:
                        print(f"Warning: No categorical features info found for LGBM model: {lgbm_model_path}. This might cause issues.")

                    models[target_mode][f"lgbm_{horse_info}"] = {
                        "model": lgbm_model,
                        "expected_columns": lgbm_feature_columns,
                        "categorical_features_with_categories": lgbm_categorical_features_with_categories, # 追加
                        "model_path": lgbm_model_path # Add model_path
                    }
                    print(f"Loaded LGBM model: {lgbm_model_path}")
                except Exception as e:
                    print(f"Error loading LGBM model {lgbm_model_path}: {e}")
            else:
                print(f"LGBM model not found: {lgbm_model_path}")

        # Load CNN model (always with horse info)
        print(f"DEBUG: Attempting to load CNN model for target_mode: {target_mode}")
        cnn_model_path = get_model_path("cnn", target_mode, "included")
        print(f"DEBUG: CNN model path: {cnn_model_path}")
        if os.path.exists(cnn_model_path):
            try:
                cnn_model = tf.keras.models.load_model(cnn_model_path, custom_objects={'FocalLoss': FocalLoss})
                cnn_flat_features_path = cnn_model_path + ".flat_features.json"
                cnn_imputation_values_path = cnn_model_path + ".imputation_values.json"
                
                cnn_flat_features = None
                if os.path.exists(cnn_flat_features_path):
                    with open(cnn_flat_features_path, "r") as f:
                        cnn_flat_features = json.load(f)
                
                cnn_imputation_values = None
                if os.path.exists(cnn_imputation_values_path):
                    with open(cnn_imputation_values_path, "r") as f:
                        cnn_imputation_values = json.load(f)

                models[target_mode]["cnn_included"] = {
                    "model": cnn_model,
                    "flat_features": cnn_flat_features,
                    "imputation_values": cnn_imputation_values,
                    "model_path": cnn_model_path # Add model_path
                }
                print(f"Loaded CNN model: {cnn_model_path}")
            except Exception as e:
                print(f"Error loading CNN model {cnn_model_path}: {e}")
        else:
            print(f"CNN model not found: {cnn_model_path}")
            
    return models
