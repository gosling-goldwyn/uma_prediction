import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    Flatten,
    Dense,
    Dropout,
    concatenate,
    BatchNormalization,
    Activation,
)
from tensorflow.keras.utils import to_categorical
import lightgbm as lgb
from tensorflow.keras import backend as K
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.isotonic import IsotonicRegression


# Focal Loss as a Keras Loss class
class FocalLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        gamma=2.0,
        alpha=None,
        name="focal_loss",
        reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
    ):
        super().__init__(name=name, reduction=reduction)
        self.gamma = gamma
        self.alpha = alpha  # alpha can be a scalar or a tensor of weights

    def call(self, y_true, y_pred):
        # Apply softmax to y_pred if it's logits
        y_pred = tf.nn.softmax(y_pred)

        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)

        # Calculate cross entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate p_t (probability of the true class)
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)

        # Calculate (1 - p_t)^gamma
        focal_term = K.pow(1.0 - p_t, self.gamma)

        # Apply alpha weighting
        if self.alpha is not None:
            alpha_factor = self.alpha * y_true
            alpha_factor = tf.reduce_sum(alpha_factor, axis=-1, keepdims=True)
        else:
            alpha_factor = 1.0

        # Combine focal term and cross entropy
        loss = focal_term * cross_entropy * alpha_factor

        # Sum over classes for each sample
        loss = K.sum(loss, axis=-1)

        return loss

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma, "alpha": self.alpha})
        return config


from scripts.data_preprocessing.lgbm_categorical_processor import (
    save_lgbm_categorical_features,
)


# GPUが利用可能か確認し、設定を行う

physical_devices = tf.config.list_physical_devices("GPU")
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU is available and memory growth is enabled.")
    except:
        print("Could not set GPU memory growth.")
else:
    print("No GPU devices found. Using CPU.")

# --- Model Paths ---
TRAINING_STATUS_FILE = "data/training_status.json"


def get_model_path(model_type, target_mode, horse_info="included", fine_tune_suffix=""):
    horse_info_suffix = ""
    if model_type != "cnn":
        horse_info_suffix = "" if horse_info == "included" else "_no_horse_info"

    model_suffix = "_with_cat" if model_type == "cnn" else horse_info_suffix

    extension = (
        "keras" if model_type == "cnn" else "pkl" if model_type == "rf" else "txt"
    )

    return f"models/{model_type}_uma_prediction_model_{target_mode}{model_suffix}{fine_tune_suffix}.{extension}"


def update_training_status(status_data):
    os.makedirs(os.path.dirname(TRAINING_STATUS_FILE), exist_ok=True)
    with open(TRAINING_STATUS_FILE, "w") as f:
        json.dump(status_data, f, indent=4)


def preprocess_data(
    df, model_type="rf", target_mode="default", exclude_horse_info=False
):
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df.dropna(subset=["rank"], inplace=True)

    # Convert 'l_days' to numeric, coercing errors to NaN
    df["l_days"] = pd.to_numeric(df["l_days"], errors="coerce")

    def assign_target(rank):
        if target_mode == "default":
            if rank == 1:
                return 0
            elif 2 <= rank <= 3:
                return 1
            else:
                return 2
        elif target_mode == "top3":
            if 1 <= rank <= 3:
                return 0
            else:
                return 1
        else:
            raise ValueError(f"Unknown target_mode: {target_mode}")

    df["target"] = df["rank"].apply(assign_target)

    # Feature Engineering
    rank_cols = ["1st_rank", "2nd_rank", "3rd_rank", "4th_rank", "5th_rank"]
    speed_idx_cols = [
        "1st_speed_idx",
        "2nd_speed_idx",
        "3rd_speed_idx",
        "4th_speed_idx",
        "5th_speed_idx",
    ]

    # Time-series features
    df["avg_rank_past_5"] = df[rank_cols].mean(axis=1)
    df["win_rate_past_5"] = (df[rank_cols] == 1).sum(axis=1) / df[
        rank_cols
    ].notna().sum(axis=1)
    df["top_3_rate_past_5"] = (df[rank_cols] <= 3).sum(axis=1) / df[
        rank_cols
    ].notna().sum(axis=1)
    df["avg_speed_idx_past_5"] = df[speed_idx_cols].mean(axis=1)
    df["max_speed_idx_past_5"] = df[speed_idx_cols].max(axis=1)
    df["std_rank_past_5"] = df[rank_cols].std(axis=1)

    # Interaction features
    df["jockey_place"] = df["jockey"].astype(str) + "_" + df["place"].astype(str)
    df["horse_place"] = df["horse_name"].astype(str) + "_" + df["place"].astype(str)
    df["horse_dist"] = df["horse_name"].astype(str) + "_" + df["dist"].astype(str)
    df["place_dist"] = df["place"].astype(str) + "_" + df["dist"].astype(str)

    # Domain knowledge features
    df["weight_ratio"] = df["weight_carry"] / df["horse_weight"]

    base_exclude_cols = ["prize", "rank", "time", "target", "date"]
    if exclude_horse_info:
        base_exclude_cols.extend(["horse_num", "horse_weight", "weight_change"])

    X = df.drop(columns=base_exclude_cols, errors="ignore")
    y = df["target"]

    # Fill missing values for all types
    imputation_map = {}
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            value = X[col].median() if not X[col].isnull().all() else 0
            X[col] = X[col].fillna(value)
            imputation_map[col] = value
        else:
            value = X[col].mode()[0] if not X[col].isnull().all() else "missing"
            X[col] = X[col].fillna(value)
            imputation_map[col] = value

    if model_type == "rf":
        high_cardinality_features = [
            "horse_name",
            "jockey",
            "jockey_place",
            "horse_place",
            "horse_dist",
        ]
        target_maps = {}

        # Target Encoding for high cardinality features
        for col in high_cardinality_features:
            if col in X.columns:
                target_map = df.groupby(col)["target"].mean()
                X[col] = X[col].map(target_map).fillna(df["target"].mean())
                target_maps[col] = target_map.to_dict()

        # One-Hot Encoding for low cardinality features
        low_cardinality_features = [
            col
            for col in X.columns
            if X[col].dtype == "object" and col not in high_cardinality_features
        ]
        X = pd.get_dummies(X, columns=low_cardinality_features, dummy_na=False)
        return X, y, target_maps, imputation_map

    elif model_type == "lgbm":
        categorical_features = [col for col in X.columns if X[col].dtype == "object"]
        # カテゴリカル特徴量のカテゴリを保持
        categorical_features_with_categories = {}
        for col in categorical_features:
            # Handle '○○' in jockey column specifically
            if col == 'jockey':
                X[col] = X[col].replace('○○', 'missing')

            # Unify missing/empty values before creating categories
            X[col] = X[col].replace('', np.nan) # Treat empty strings as NaN
            X[col] = X[col].fillna("missing")   # Fill all NaN values with 'missing'

            # Create a sorted list of categories to ensure consistency
            all_categories = X[col].unique().tolist()
            if 'missing' not in all_categories:
                all_categories.append('missing')
            all_categories = sorted(all_categories) # Sort the final list

            X[col] = pd.Categorical(X[col], categories=all_categories)
            categorical_features_with_categories[col] = all_categories

        # print(f"Categorical features with categories for LGBM: {categorical_features_with_categories}")
        return X, y, categorical_features_with_categories, imputation_map

    elif model_type == "cnn":
        sequence_features = [
            "1st_speed_idx",
            "2nd_speed_idx",
            "3rd_speed_idx",
            "4th_speed_idx",
            "5th_speed_idx",
            "1st_lead_idx",
            "2nd_lead_idx",
            "3rd_lead_idx",
            "4th_lead_idx",
            "5th_lead_idx",
            "1st_pace_idx",
            "2nd_pace_idx",
            "3rd_pace_idx",
            "4th_pace_idx",
            "5th_pace_idx",
            "1st_rising_idx",
            "2nd_rising_idx",
            "3rd_rising_idx",
            "4th_rising_idx",
            "5th_rising_idx",
        ]
        X_seq_df = df[sequence_features].copy()
        imputation_values = {}
        for col in sequence_features:
            median_val = (
                X_seq_df[col].median() if not X_seq_df[col].isnull().all() else 0
            )
            X_seq_df[col] = pd.to_numeric(X_seq_df[col], errors="coerce").fillna(
                median_val
            )
            imputation_values[col] = median_val
        X_seq = X_seq_df.values.reshape(len(df), 5, len(sequence_features) // 5)

        flat_numerical_features = [
            "age",
            "weight_carry",
            "horse_num",
            "horse_weight",
            "weight_change",
        ]
        flat_categorical_features = ["sex", "jockey", "field", "weather", "place"]

        X_flat_num = df[flat_numerical_features].copy()
        for col in X_flat_num.columns:
            median_val = (
                X_flat_num[col].median() if not X_flat_num[col].isnull().all() else 0
            )
            X_flat_num[col] = pd.to_numeric(X_flat_num[col], errors="coerce").fillna(
                median_val
            )
            imputation_values[col] = median_val

        X_flat_cat = df[flat_categorical_features].copy()
        for col in X_flat_cat.columns:
            mode_val = X_flat_cat[col].mode()[0]
            X_flat_cat[col] = X_flat_cat[col].astype(str).fillna(mode_val)
            imputation_values[col] = mode_val
        X_flat_cat_dummies = pd.get_dummies(
            X_flat_cat, columns=flat_categorical_features
        )

        X_flat = pd.concat([X_flat_num, X_flat_cat_dummies], axis=1)
        y_cnn = to_categorical(df["target"], num_classes=3)

        # Calculate class weights
        from sklearn.utils.class_weight import compute_class_weight

        class_weights = compute_class_weight(
            class_weight="balanced", classes=np.unique(df["target"]), y=df["target"]
        )
        class_weight_dict = dict(zip(np.unique(df["target"]), class_weights))

        return (
            [X_seq, X_flat],
            y_cnn,
            X_flat.columns.tolist(),
            imputation_values, # Note: This is different from imputation_map, specific to CNN
            class_weight_dict,
            imputation_map, # Return the general imputation map as well
        )

    else:
        raise ValueError("Invalid model_type specified.")


def build_multi_input_cnn_model(
    sequence_input_shape, flat_input_shape, num_classes, class_weight_dict, params=None
):
    cnn_params = {
        "epochs": 10,
        "batch_size": 32,
        "conv1d_filters": 32,
        "dense_units": 128,
    }
    if params:
        cnn_params.update(params)

    seq_input = Input(shape=sequence_input_shape, name="sequence_input")
    x1 = Conv1D(filters=cnn_params["conv1d_filters"], kernel_size=2, activation="relu")(
        seq_input
    )
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling1D(pool_size=2)(x1)
    x1 = Dropout(0.25)(x1)
    x1 = Conv1D(
        filters=cnn_params["conv1d_filters"] * 2, kernel_size=2, activation="relu"
    )(x1)  # Double filters for second conv layer
    x1 = BatchNormalization()(x1)
    x1 = Flatten()(x1)

    flat_input = Input(shape=flat_input_shape, name="flat_input")
    x2 = Dense(cnn_params["dense_units"] // 4, activation="relu")(
        flat_input
    )  # Use a quarter of dense_units for first dense layer
    x2 = BatchNormalization()(x2)

    concatenated = concatenate([x1, x2])
    final_output = Dense(cnn_params["dense_units"], activation="relu")(concatenated)
    final_output = BatchNormalization()(final_output)
    final_output = Dropout(0.25)(final_output)
    final_output = Dense(num_classes)(final_output)  # No activation here

    model = Model(inputs=[seq_input, flat_input], outputs=final_output)

    # Convert class_weight_dict to a tensor for FocalLoss alpha
    # Ensure the order of weights matches the class indices (0, 1, 2 for default; 0, 1 for top3)
    # Fill with 1.0 if a class is missing in class_weight_dict (shouldn't happen if balanced)
    alpha_list = [class_weight_dict.get(i, 1.0) for i in range(num_classes)]
    alpha_tensor = tf.constant(alpha_list, dtype=tf.float32)

    model.compile(
        optimizer="adam", loss=FocalLoss(alpha=alpha_tensor), metrics=["accuracy"]
    )
    return model, cnn_params["epochs"], cnn_params["batch_size"]


def train_model(
    model_type,
    X,
    y,
    target_mode,
    horse_info="included",
    hf_api=None,
    hf_token=None,
    hf_model_repo_id=None,
    params=None,
    **kwargs,
):
    print(
        f"\n--- Training {model_type.upper()} ({target_mode}, horse_info: {horse_info}) ---"
    )
    if (isinstance(X, list) and (X[0].size == 0 or X[1].size == 0)) or (
        not isinstance(X, list) and X.empty
    ):
        print(f"No data to train the {model_type} model. Exiting training.")
        return 0.0

    model_path = get_model_path(
        model_type,
        target_mode,
        horse_info,
        fine_tune_suffix=kwargs.get("fine_tune_suffix", ""),
    )
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    if model_type == "cnn":
        X_train_seq, X_test_seq, X_train_flat, X_test_flat, y_train, y_test = (
            train_test_split(
                X[0],
                X[1],
                y,
                test_size=0.2,
                random_state=42,
                stratify=np.argmax(y, axis=1),
            )
        )
        X_train = [X_train_seq, X_train_flat]
        X_test = [X_test_seq, X_test_flat]

        model, epochs, batch_size = build_multi_input_cnn_model(
            (X[0].shape[1], X[0].shape[2]),
            (X[1].shape[1],),
            y.shape[1],
            kwargs.get("class_weight_dict"),
            params=params,
        )
        model.summary()

        model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1,
        )
        model.save(model_path)
        with open(model_path + ".flat_features.json", "w") as f:
            json.dump(kwargs["flat_features_columns"], f)
        with open(model_path + ".imputation.json", "w") as f:
            json.dump(kwargs["imputation_map"], f)
        with open(model_path + ".imputation_values.json", "w") as f:
            json.dump(kwargs["imputation_values"], f)

        # Upload CNN model and associated files to Hugging Face if hf_api is provided
        if hf_api and hf_token and hf_model_repo_id:
            try:
                hf_api.upload_file(
                    path_or_fileobj=model_path,
                    path_in_repo=os.path.basename(model_path),
                    repo_id=hf_model_repo_id,
                    repo_type="model",
                    token=hf_token,
                )
                hf_api.upload_file(
                    path_or_fileobj=model_path + ".flat_features.json",
                    path_in_repo=os.path.basename(model_path) + ".flat_features.json",
                    repo_id=hf_model_repo_id,
                    repo_type="model",
                    token=hf_token,
                )
                hf_api.upload_file(
                    path_or_fileobj=model_path + ".imputation_values.json",
                    path_in_repo=os.path.basename(model_path)
                    + ".imputation_values.json",
                    repo_id=hf_model_repo_id,
                    repo_type="model",
                    token=hf_token,
                )
                print(
                    f"Uploaded CNN model and metadata to Hugging Face: {hf_model_repo_id}"
                )
            except Exception as e:
                print(f"Error uploading CNN model to Hugging Face: {e}")

        # Calculate metrics for Optuna and saving
        y_pred_proba = model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true_labels = np.argmax(y_test, axis=1)

        accuracy = np.mean(y_pred == y_true_labels)
        f1 = f1_score(y_true_labels, y_pred, average="weighted")
        try:
            # roc_auc_score requires at least two classes present in y_true
            if len(np.unique(y_true_labels)) > 1:
                auc_roc = roc_auc_score(
                    y_true_labels, y_pred_proba, multi_class="ovr", average="weighted"
                )
            else:
                auc_roc = None
        except ValueError:
            auc_roc = None

        metrics = {"accuracy": accuracy, "f1_score": f1, "auc_roc_score": auc_roc}
        with open(model_path + ".metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        print(
            f"CNN Model Test Accuracy: {accuracy}, F1 Score: {f1}, AUC-ROC Score: {auc_roc}"
        )

        if hf_api and hf_token and hf_model_repo_id:
            try:
                hf_api.upload_file(
                    path_or_fileobj=model_path + ".metrics.json",
                    path_in_repo=os.path.basename(model_path) + ".metrics.json",
                    repo_id=hf_model_repo_id,
                    repo_type="model",
                    token=hf_token,
                )
                print(f"Uploaded CNN model metrics to Hugging Face: {hf_model_repo_id}")
            except Exception as e:
                print(f"Error uploading CNN model metrics to Hugging Face: {e}")

        return accuracy

    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        if model_type == "rf":
            rf_params = {
                "n_estimators": 100,
                "random_state": 42,
                "class_weight": "balanced",
                "max_depth": None,  # Default for RandomForestClassifier
                "min_samples_split": 2,
                "min_samples_leaf": 1,
            }
            if params:
                # Remove target_mode from params if it exists, as RandomForestClassifier does not accept it
                filtered_params = {
                    k: v for k, v in params.items() if k != "target_mode"
                }
                rf_params.update(filtered_params)
            model = RandomForestClassifier(**rf_params)
            model.fit(X_train, y_train)

            # Calibrate probabilities
            y_pred_proba = model.predict_proba(X_test)
            calibrators = []
            for i in range(y_pred_proba.shape[1]):
                iso_reg = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
                y_test_class = (y_test == i).astype(int)
                iso_reg.fit(y_pred_proba[:, i], y_test_class)
                calibrators.append(iso_reg)

            # Save model and calibrators
            joblib.dump({"model": model, "calibrators": calibrators}, model_path)
            with open(model_path + ".feature_columns.json", "w") as f:
                json.dump(X_train.columns.tolist(), f)
            with open(model_path + ".imputation.json", "w") as f:
                json.dump(kwargs["imputation_map"], f)
            if "target_maps" in kwargs:
                with open(model_path + ".target_maps.json", "w") as f:
                    json.dump(kwargs["target_maps"], f)

            # Upload RF model and associated files to Hugging Face if hf_api is provided
            if hf_api and hf_token and hf_model_repo_id:
                try:
                    hf_api.upload_file(
                        path_or_fileobj=model_path,
                        path_in_repo=os.path.basename(model_path),
                        repo_id=hf_model_repo_id,
                        repo_type="model",
                        token=hf_token,
                    )
                    hf_api.upload_file(
                        path_or_fileobj=model_path + ".feature_columns.json",
                        path_in_repo=os.path.basename(model_path)
                        + ".feature_columns.json",
                        repo_id=hf_model_repo_id,
                        repo_type="model",
                        token=hf_token,
                    )
                    if "target_maps" in kwargs:
                        hf_api.upload_file(
                            path_or_fileobj=model_path + ".target_maps.json",
                            path_in_repo=os.path.basename(model_path)
                            + ".target_maps.json",
                            repo_id=hf_model_repo_id,
                            repo_type="model",
                            token=hf_token,
                        )
                    print(
                        f"Uploaded RF model and metadata to Hugging Face: {hf_model_repo_id}"
                    )
                except Exception as e:
                    print(f"Error uploading RF model to Hugging Face: {e}")

            # Calculate metrics for Optuna and saving
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)

            accuracy = np.mean(y_pred == y_test)
            f1 = f1_score(y_test, y_pred, average="weighted")
            try:
                if len(np.unique(y_test)) > 1:
                    auc_roc = roc_auc_score(
                        y_test, y_pred_proba, multi_class="ovr", average="weighted"
                    )
                else:
                    auc_roc = None
            except ValueError:
                auc_roc = None

            metrics = {"accuracy": accuracy, "f1_score": f1, "auc_roc_score": auc_roc}
            with open(model_path + ".metrics.json", "w") as f:
                json.dump(metrics, f, indent=4)

            print(
                f"RF Model Test Accuracy: {accuracy}, F1 Score: {f1}, AUC-ROC Score: {auc_roc}"
            )

            if hf_api and hf_token and hf_model_repo_id:
                try:
                    hf_api.upload_file(
                        path_or_fileobj=model_path + ".metrics.json",
                        path_in_repo=os.path.basename(model_path) + ".metrics.json",
                        repo_id=hf_model_repo_id,
                        repo_type="model",
                        token=hf_token,
                    )
                    print(
                        f"Uploaded RF model metrics to Hugging Face: {hf_model_repo_id}"
                    )
                except Exception as e:
                    print(f"Error uploading RF model metrics to Hugging Face: {e}")

            return accuracy

        elif model_type == "lgbm":
            if kwargs.get("base_model_path"):
                base_model_path = kwargs.get("base_model_path")
                print(f"Loading base LGBM model from {base_model_path} for fine-tuning...")
                model = joblib.load(base_model_path)
                # For fine-tuning, we just continue fitting the loaded model
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_test, y_test)],
                    categorical_feature=kwargs["categorical_features"],
                    callbacks=[lgb.early_stopping(10, verbose=True)],
                )
            else:
                lgbm_params = {
                    "objective": "multiclass",
                    "num_class": len(np.unique(y_train)),
                    "metric": "multi_logloss",
                    "boosting_type": "gbdt",
                    "n_jobs": -1,
                    "seed": 42,
                    "verbose": -1,
                    "class_weight": "balanced",
                }
                if params:
                    lgbm_params.update(params)

                model = lgb.LGBMClassifier(**lgbm_params)
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_test, y_test)],
                    categorical_feature=kwargs["categorical_features"],
                    callbacks=[lgb.early_stopping(10, verbose=True)],
                )
            joblib.dump(model, model_path)
            with open(model_path + ".feature_columns.json", "w") as f:
                json.dump(X_train.columns.tolist(), f)
            with open(model_path + ".imputation.json", "w") as f:
                json.dump(kwargs["imputation_map"], f)
            # カテゴリカル特徴量のカテゴリ情報を保存
            if "categorical_features_with_categories" in kwargs:
                save_lgbm_categorical_features(
                    model_path, kwargs["categorical_features_with_categories"]
                )

            # Upload LGBM model and associated files to Hugging Face if hf_api is provided
            if hf_api and hf_token and hf_model_repo_id:
                try:
                    hf_api.upload_file(
                        path_or_fileobj=model_path,
                        path_in_repo=os.path.basename(model_path),
                        repo_id=hf_model_repo_id,
                        repo_type="model",
                        token=hf_token,
                    )
                    hf_api.upload_file(
                        path_or_fileobj=model_path + ".feature_columns.json",
                        path_in_repo=os.path.basename(model_path)
                        + ".feature_columns.json",
                        repo_id=hf_model_repo_id,
                        repo_type="model",
                        token=hf_token,
                    )
                    if "categorical_features_with_categories" in kwargs:
                        hf_api.upload_file(
                            path_or_fileobj=model_path + ".categorical_features.json",
                            path_in_repo=os.path.basename(model_path)
                            + ".categorical_features.json",
                            repo_id=hf_model_repo_id,
                            repo_type="model",
                            token=hf_token,
                        )
                    print(
                        f"Uploaded LGBM model and metadata to Hugging Face: {hf_model_repo_id}"
                    )
                except Exception as e:
                    print(f"Error uploading LGBM model to Hugging Face: {e}")

            # Calculate metrics for Optuna and saving
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)

            accuracy = np.mean(y_pred == y_test)
            f1 = f1_score(y_test, y_pred, average="weighted")
            try:
                if len(np.unique(y_test)) > 1:
                    auc_roc = roc_auc_score(
                        y_test, y_pred_proba, multi_class="ovr", average="weighted"
                    )
                else:
                    auc_roc = None
            except ValueError:
                auc_roc = None

            metrics = {"accuracy": accuracy, "f1_score": f1, "auc_roc_score": auc_roc}
            with open(model_path + ".metrics.json", "w") as f:
                json.dump(metrics, f, indent=4)

            print(
                f"LGBM Model Test Accuracy: {accuracy}, F1 Score: {f1}, AUC-ROC Score: {auc_roc}"
            )

            if hf_api and hf_token and hf_model_repo_id:
                try:
                    hf_api.upload_file(
                        path_or_fileobj=model_path + ".metrics.json",
                        path_in_repo=os.path.basename(model_path) + ".metrics.json",
                        repo_id=hf_model_repo_id,
                        repo_type="model",
                        token=hf_token,
                    )
                    print(
                        f"Uploaded LGBM model metrics to Hugging Face: {hf_model_repo_id}"
                    )
                except Exception as e:
                    print(f"Error uploading LGBM model metrics to Hugging Face: {e}")

            return accuracy

    print(f"{model_type.upper()} Model saved to {model_path}")
    return None
