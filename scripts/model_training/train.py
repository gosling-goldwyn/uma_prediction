import pandas as pd
import json
import sys

from scripts.model_training.training_utils import (
    get_model_path,
    update_training_status,
    preprocess_data,
    train_model,
)

# --- Model Paths ---
DATA_FILE = "data/index_dataset.parquet"

def load_data():
    try:
        return pd.read_parquet(DATA_FILE)
    except Exception as e:
        print(f"Error loading {DATA_FILE}: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    update_training_status(
        {"status": "running", "message": "Starting training process..."}
    )
    df = load_data()
    if df.empty:
        update_training_status({"status": "error", "message": "No data loaded."})
        sys.exit()

    for target_mode in ["default", "top3"]:
        # With horse info
        X_rf, y_rf, target_maps_rf = preprocess_data(
            df.copy(), model_type="rf", target_mode=target_mode
        )
        train_model("rf", X_rf, y_rf, target_mode, target_maps=target_maps_rf)
        X_lgbm, y_lgbm, cats_lgbm_with_categories = preprocess_data( # 変数名を変更
            df.copy(), model_type="lgbm", target_mode=target_mode
        )
        train_model("lgbm", X_lgbm, y_lgbm, target_mode,
                    categorical_features=[col for col in cats_lgbm_with_categories.keys()], # LGBMの引数にはカラム名リストを渡す
                    categorical_features_with_categories=cats_lgbm_with_categories)

        # WITHOUT horse info
        X_rf_no, y_rf_no, target_maps_rf_no = preprocess_data(
            df.copy(), model_type="rf", target_mode=target_mode, exclude_horse_info=True
        )
        train_model(
            "rf", X_rf_no, y_rf_no, target_mode, horse_info="excluded", target_maps=target_maps_rf_no
        )
        X_lgbm_no, y_lgbm_no, cats_lgbm_no_with_categories = preprocess_data( # 変数名を変更
            df.copy(),
            model_type="lgbm",
            target_mode=target_mode,
            exclude_horse_info=True,
        )
        train_model("lgbm", X_lgbm_no, y_lgbm_no, target_mode,
                    categorical_features=[col for col in cats_lgbm_no_with_categories.keys()], # LGBMの引数にはカラム名リストを渡す
                    categorical_features_with_categories=cats_lgbm_no_with_categories, horse_info="excluded")

        # CNN with categorical features
        X_cnn, y_cnn, flat_cols, imputation_values, class_weight_dict = preprocess_data(
            df.copy(), model_type="cnn", target_mode=target_mode
        )
        train_model(
            "cnn",
            X_cnn,
            y_cnn,
            target_mode,
            flat_features_columns=flat_cols,
            imputation_values=imputation_values,
            class_weight_dict=class_weight_dict,
        )

    print("Model training finished.")
    update_training_status(
        {"status": "completed", "message": "All models trained successfully."}
    )