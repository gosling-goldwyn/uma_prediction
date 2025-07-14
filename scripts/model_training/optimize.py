import optuna
from optuna.integration import TensorBoardCallback
import pandas as pd
import os
import numpy as np

from scripts.model_training.training_utils import preprocess_data, train_model

DATA_FILE = "data/index_dataset.parquet"

def load_data():
    try:
        return pd.read_parquet(DATA_FILE)
    except Exception as e:
        print(f"Error loading {DATA_FILE}: {e}")
        return pd.DataFrame()

def objective(trial):
    df = load_data()
    if df.empty:
        raise ValueError("No data loaded for Optuna objective.")

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
        # CNNモデルのtrain_modelは直接accuracyを返さないため、別途評価
        # train_model内でvalidation_dataのaccuracyを返すように修正が必要
        # 現状はNoneが返るため、ここでは0.0を返す
        # TODO: train_model for CNN should return accuracy
        accuracy = 0.0 # Placeholder for now

    return accuracy

if __name__ == "__main__":
    # Create a directory for TensorBoard logs
    log_dir = "logs/optuna"
    os.makedirs(log_dir, exist_ok=True)

    # Create a TensorBoard callback
    tb_callback = TensorBoardCallback(log_dir, metric_name="accuracy")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10, callbacks=[tb_callback])

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Save best parameters for each model type
    best_params_dir = "params"
    os.makedirs(best_params_dir, exist_ok=True)

    # Group trials by model_type
    trials_by_model_type = {}
    for t in study.trials:
        if "model_type" in t.params:
            model_type = t.params["model_type"]
            if model_type not in trials_by_model_type or t.value > trials_by_model_type[model_type]["value"]:
                trials_by_model_type[model_type] = {"value": t.value, "params": t.params}

    for model_type, data in trials_by_model_type.items():
        params_to_save = {k: v for k, v in data["params"].items() if not k.startswith(f"{model_type}_")}
        # Extract model-specific params
        model_specific_params = {}
        for k, v in data["params"].items():
            if k.startswith(f"{model_type}_"):
                model_specific_params[k.replace(f"{model_type}_", "")] = v
        
        # Combine model-specific params with other relevant params (e.g., target_mode)
        final_params = {"target_mode": data["params"].get("target_mode", "default")} # Assuming default for now
        final_params.update(model_specific_params)

        param_file_path = os.path.join(best_params_dir, f"best_params_{model_type}.json")
        with open(param_file_path, "w") as f:
            json.dump(final_params, f, indent=4)
        print(f"Saved best parameters for {model_type} to {param_file_path}")