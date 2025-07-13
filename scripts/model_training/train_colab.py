import pandas as pd
import json
import sys
import os
import argparse

from datasets import load_dataset
from huggingface_hub import HfApi, create_repo
from dotenv import load_dotenv

from scripts.model_training.training_utils import (
    get_model_path,
    update_training_status,
    preprocess_data,
    train_model,
)

# .envファイルから環境変数を読み込む
load_dotenv()

# Hugging Faceリポジトリの設定
HF_MODEL_REPO_ID = os.getenv("REPO_ID")
hf_token = os.getenv("HF_TOKEN")

if hf_token is None:
    print(
        "Hugging Face token (HF_TOKEN) not found in environment variables or .env file."
    )
    print(
        "Please ensure .env file exists and contains HF_TOKEN, or set it as an environment variable."
    )
    sys.exit()

hf_api = HfApi(token=hf_token)

# --- Model Paths ---
DATASET_REPO_ID = os.getenv("DATASET_REPO_ID")


def load_data():
    try:
        # Hugging Face Datasetsからデータを読み込む
        dataset = load_dataset(DATASET_REPO_ID, split="train")
        # Pandas DataFrameに変換
        df = dataset.to_pandas()
        print(f"Successfully loaded data from Hugging Face Dataset: {DATASET_REPO_ID}")
        return df
    except Exception as e:
        print(f"Error loading data from Hugging Face Dataset: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train prediction models.")
    parser.add_argument('--model_type', type=str, default='all',
                        choices=['rf', 'lgbm', 'cnn', 'all'],
                        help='Type of model to train (rf, lgbm, cnn, or all)')
    args = parser.parse_args()
    model_to_train = args.model_type

    # Create model repository on Hugging Face if it doesn't exist
    try:
        create_repo(
            repo_id=HF_MODEL_REPO_ID, repo_type="model", exist_ok=True, token=hf_token
        )
        print(f"Model repository '{HF_MODEL_REPO_ID}' created or already exists.")
    except Exception as e:
        print(f"Error creating/checking model repository: {e}")
        sys.exit()

    update_training_status(
        {"status": "running", "message": f"Starting training process for {model_to_train}..."}
    )
    df = load_data()
    if df.empty:
        update_training_status({"status": "error", "message": "No data loaded."})
        sys.exit()

    for target_mode in ["default", "top3"]:
        if model_to_train in ['rf', 'all']:
            # With horse info
            X_rf, y_rf, target_maps_rf = preprocess_data(
                df.copy(), model_type="rf", target_mode=target_mode
            )
            train_model(
                "rf",
                X_rf,
                y_rf,
                target_mode,
                hf_api=hf_api,
                hf_token=hf_token,
                hf_model_repo_id=HF_MODEL_REPO_ID,
                target_maps=target_maps_rf,
            )

            # WITHOUT horse info
            X_rf_no, y_rf_no, target_maps_rf_no = preprocess_data(
                df.copy(), model_type="rf", target_mode=target_mode, exclude_horse_info=True
            )
            train_model(
                "rf",
                X_rf_no,
                y_rf_no,
                target_mode,
                horse_info="excluded",
                hf_api=hf_api,
                hf_token=hf_token,
                hf_model_repo_id=HF_MODEL_REPO_ID,
                target_maps=target_maps_rf_no,
            )

        if model_to_train in ['lgbm', 'all']:
            # With horse info
            X_lgbm, y_lgbm, cats_lgbm_with_categories = preprocess_data(
                df.copy(), model_type="lgbm", target_mode=target_mode
            )
            train_model(
                "lgbm",
                X_lgbm,
                y_lgbm,
                target_mode,
                hf_api=hf_api,
                hf_token=hf_token,
                hf_model_repo_id=HF_MODEL_REPO_ID,
                categorical_features=[
                    col for col in cats_lgbm_with_categories.keys()
                ],
                categorical_features_with_categories=cats_lgbm_with_categories,
            )

            # WITHOUT horse info
            X_lgbm_no, y_lgbm_no, cats_lgbm_no_with_categories = (
                preprocess_data(
                    df.copy(),
                    model_type="lgbm",
                    target_mode=target_mode,
                    exclude_horse_info=True,
                )
            )
            train_model(
                "lgbm",
                X_lgbm_no,
                y_lgbm_no,
                target_mode,
                hf_api=hf_api,
                hf_token=hf_token,
                hf_model_repo_id=HF_MODEL_REPO_ID,
                categorical_features=[
                    col for col in cats_lgbm_no_with_categories.keys()
                ],
                categorical_features_with_categories=cats_lgbm_no_with_categories,
                horse_info="excluded",
            )

        if model_to_train in ['cnn', 'all']:
            # CNN with categorical features
            X_cnn, y_cnn, flat_cols, imputation_values, class_weight_dict = preprocess_data(
                df.copy(), model_type="cnn", target_mode=target_mode
            )
            train_model(
                "cnn",
                X_cnn,
                y_cnn,
                target_mode,
                hf_api=hf_api,
                hf_token=hf_token,
                hf_model_repo_id=HF_MODEL_REPO_ID,
                flat_features_columns=flat_cols,
                imputation_values=imputation_values,
                class_weight_dict=class_weight_dict,
            )

    print(f"Model training for {model_to_train} finished.")
    update_training_status(
        {"status": "completed", "message": f"{model_to_train.upper()} models trained successfully."}
    )
