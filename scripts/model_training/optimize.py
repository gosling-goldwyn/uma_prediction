import optuna
from optuna.integration import TensorBoardCallback
import pandas as pd
import os
import numpy as np
import json
import argparse # Added for argparse

from scripts.model_training.training_utils import preprocess_data, train_model

DATA_FILE = "data/index_dataset.parquet"

def load_data():
    try:
        return pd.read_parquet(DATA_FILE)
    except Exception as e:
        print(f"Error loading {DATA_FILE}: {e}")
        return pd.DataFrame()

def objective(trial, model_type_filter=None):
    df = load_data()
    if df.empty:
        raise ValueError("No data loaded for Optuna objective.")

    # If model_type_filter is provided, use it directly
    if model_type_filter:
        model_type = model_type_filter
    else:
        # Otherwise, let Optuna suggest the model type
        model_type = trial.suggest_categorical("model_type", ["rf", "lgbm", "cnn"])
    
    target_mode = "default" # For simplicity, optimize only default target mode for now

    accuracy = 0.0

    if model_type == "rf":
        rf_params = {
            "n_estimators": trial.suggest_int("rf_n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("rf_max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("rf_min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 1, 10),
        }
        X_rf, y_rf, target_maps_rf = preprocess_data(
            df.copy(), model_type="rf", target_mode=target_mode
        )
        accuracy = train_model(
            "rf", X_rf, y_rf, target_mode, target_maps=target_maps_rf, params=rf_params
        )

    elif model_type == "lgbm":
        lgbm_params = {
            "n_estimators": trial.suggest_int("lgbm_n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("lgbm_learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("lgbm_num_leaves", 20, 100),
        }
        X_lgbm, y_lgbm, cats_lgbm_with_categories = preprocess_data(
            df.copy(), model_type="lgbm", target_mode=target_mode
        )
        accuracy = train_model(
            "lgbm",
            X_lgbm,
            y_lgbm,
            target_mode,
            categorical_features=[col for col in cats_lgbm_with_categories.keys()],
            categorical_features_with_categories=cats_lgbm_with_categories,
            params=lgbm_params,
        )

    elif model_type == "cnn":
        cnn_params = {
            "epochs": trial.suggest_int("cnn_epochs", 5, 20),
            "batch_size": trial.suggest_categorical("cnn_batch_size", [16, 32, 64]),
            "conv1d_filters": trial.suggest_categorical("cnn_conv1d_filters", [32, 64, 128]),
            "dense_units": trial.suggest_categorical("cnn_dense_units", [64, 128, 256]),
        }
        X_cnn, y_cnn, flat_cols, imputation_values, class_weight_dict = preprocess_data(
            df.copy(), model_type="cnn", target_mode=target_mode
        )
        accuracy = train_model(
            "cnn", X_cnn, y_cnn, target_mode, 
            flat_features_columns=flat_cols, 
            imputation_values=imputation_values, 
            class_weight_dict=class_weight_dict, 
            params=cnn_params
        )

    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize hyperparameters for specified model types.")
    parser.add_argument(
        "--model-types",
        nargs="+",
        choices=["rf", "lgbm", "cnn"],
        default=["rf", "lgbm", "cnn"],
        help="Specify one or more model types to optimize (e.g., --model-types rf cnn)"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=10,
        help="Number of optimization trials per model type."
    )
    args = parser.parse_args()

    for model_type_to_run in args.model_types:
        print(f"\n--- Optimizing {model_type_to_run.upper()} model ---")
        
        log_dir = f"logs/optuna/{model_type_to_run}"
        os.makedirs(log_dir, exist_ok=True)

        tb_callback = TensorBoardCallback(log_dir, metric_name="accuracy")

        study_name = f"uma_prediction_optimization_{model_type_to_run}"
        study = optuna.create_study(
            direction="maximize", 
            storage="sqlite:///logs/optuna/optuna_study.db", 
            study_name=study_name,
            load_if_exists=True # Load existing study if it exists
        )
        
        study.optimize(
            lambda trial: objective(trial, model_type_filter=model_type_to_run), 
            n_trials=args.n_trials, 
            callbacks=[tb_callback]
        )

        print(f"Number of finished trials for {model_type_to_run}: ", len(study.trials))
        print(f"Best trial for {model_type_to_run}:")
        trial = study.best_trial

        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        best_params_dir = "params"
        os.makedirs(best_params_dir, exist_ok=True)

        model_specific_params = {}
        for k, v in trial.params.items():
            if k.startswith(f"{model_type_to_run}_"):
                model_specific_params[k.replace(f"{model_type_to_run}_", "")] = v
        
        final_params = model_specific_params

        param_file_path = os.path.join(best_params_dir, f"best_params_{model_type_to_run}.json")
        with open(param_file_path, "w") as f:
            json.dump(final_params, f, indent=4)
        print(f"Saved best parameters for {model_type_to_run} to {param_file_path}")
