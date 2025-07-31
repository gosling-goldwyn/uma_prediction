import pandas as pd
import json
import sys
import argparse
import os

from scripts.model_training.training_utils import (
    get_model_path,
    update_training_status,
    preprocess_data,
    train_model,
)

DATA_FILE = "data/index_dataset.parquet"

def load_data():
    try:
        return pd.read_parquet(DATA_FILE)
    except Exception as e:
        print(f"Error loading {DATA_FILE}: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a pre-trained model.")
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['rf', 'lgbm', 'cnn'],
                        help='Type of model to fine-tune (rf, lgbm, cnn).')
    parser.add_argument('--target_mode', type=str, default='default',
                        choices=['default', 'top3'],
                        help='Target mode for fine-tuning (default or top3).')
    parser.add_argument('--base_model_path', type=str, required=True,
                        help='Path to the base model for fine-tuning.')
    parser.add_argument('--filter_place', type=str, default=None,
                        help='Filter data by place (e.g., "東京").')
    parser.add_argument('--params_file', type=str, default=None,
                        help='Path to a JSON file containing optimized hyperparameters for fine-tuning.')
    args = parser.parse_args()

    model_to_fine_tune = args.model_type
    target_mode_to_fine_tune = args.target_mode
    base_model_path = args.base_model_path
    params = None
    if args.params_file:
        try:
            with open(args.params_file, 'r') as f:
                params = json.load(f)
            print(f"Loaded parameters from {args.params_file}: {params}")
        except Exception as e:
            print(f"Error loading parameters from {args.params_file}: {e}")
            sys.exit(1)

    update_training_status(
        {"status": "running", "message": f"Starting fine-tuning process for {model_to_fine_tune}..."}
    )
    df = load_data()
    if df.empty:
        update_training_status({"status": "error", "message": "No data loaded."})
        sys.exit()

    if not os.path.exists(base_model_path):
        update_training_status({"status": "error", "message": f"Base model not found at {base_model_path}."})
        sys.exit()

    filter_conditions = {}
    if args.filter_place:
        filter_conditions['place'] = args.filter_place

    if filter_conditions:
        for col, value in filter_conditions.items():
            if col in df.columns:
                df = df[df[col] == value]
            else:
                print(f"Warning: Filter column '{col}' not found in data.")
        if df.empty:
            update_training_status({"status": "error", "message": "No data left after filtering for fine-tuning."})
            sys.exit()
        print(f"Data filtered for fine-tuning. Remaining records: {len(df)}")
    else:
        print("No filter conditions provided. Fine-tuning on entire dataset.")

    # Determine horse_info based on base_model_path (assuming naming convention)
    horse_info = "included"
    if "_no_horse_info" in base_model_path:
        horse_info = "excluded"

    # Generate fine-tune suffix from filter_conditions
    fine_tune_suffix = ""
    if filter_conditions:
        suffix_parts = []
        for k, v in sorted(filter_conditions.items()): # Sort for consistent suffix
            suffix_parts.append(f"{k}_{v}")
        fine_tune_suffix = "_" + "_".join(suffix_parts)

    if model_to_fine_tune == 'rf':
        X_rf, y_rf, target_maps_rf, imputation_map_rf = preprocess_data(
            df.copy(), model_type="rf", target_mode=target_mode_to_fine_tune, exclude_horse_info=(horse_info=="excluded")
        )
        train_model("rf", X_rf, y_rf, target_mode_to_fine_tune, target_maps=target_maps_rf, params=params, base_model_path=base_model_path, horse_info=horse_info, fine_tune_suffix=fine_tune_suffix, imputation_map=imputation_map_rf)

    elif model_to_fine_tune == 'lgbm':
        X_lgbm, y_lgbm, cats_lgbm_with_categories, imputation_map_lgbm = preprocess_data(
            df.copy(), model_type="lgbm", target_mode=target_mode_to_fine_tune, exclude_horse_info=(horse_info=="excluded")
        )
        train_model("lgbm", X_lgbm, y_lgbm, target_mode_to_fine_tune,
                    categorical_features=[col for col in cats_lgbm_with_categories.keys()],
                    categorical_features_with_categories=cats_lgbm_with_categories, params=params, base_model_path=base_model_path, horse_info=horse_info, fine_tune_suffix=fine_tune_suffix, imputation_map=imputation_map_lgbm)

    elif model_to_fine_tune == 'cnn':
        X_cnn, y_cnn, flat_cols, imputation_values, class_weight_dict, imputation_map_cnn = preprocess_data(
            df.copy(), model_type="cnn", target_mode=target_mode_to_fine_tune
        )
        train_model(
            "cnn",
            X_cnn,
            y_cnn,
            target_mode_to_fine_tune,
            flat_features_columns=flat_cols,
            imputation_values=imputation_values,
            class_weight_dict=class_weight_dict,
            params=params,
            base_model_path=base_model_path,
            fine_tune_suffix=fine_tune_suffix,
            imputation_map=imputation_map_cnn
        )

    print(f"Fine-tuning for {model_to_fine_tune} finished.")
    update_training_status(
        {"status": "completed", "message": f"{model_to_fine_tune.upper()} model fine-tuned successfully."}
    )