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
    Activation
)
from tensorflow.keras.utils import to_categorical
import lightgbm as lgb

from scripts.data_preprocessing.lgbm_categorical_processor import save_lgbm_categorical_features

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

def get_model_path(model_type, target_mode, horse_info="included"):
    horse_info_suffix = ""
    if model_type != "cnn":
        horse_info_suffix = "" if horse_info == "included" else "_no_horse_info"

    model_suffix = "_with_cat" if model_type == "cnn" else horse_info_suffix

    extension = (
        "keras" if model_type == "cnn" else "pkl" if model_type == "rf" else "txt"
    )

    return f"models/{model_type}_uma_prediction_model_{target_mode}{model_suffix}.{extension}"

def update_training_status(status_data):
    os.makedirs(os.path.dirname(TRAINING_STATUS_FILE), exist_ok=True)
    with open(TRAINING_STATUS_FILE, "w") as f:
        json.dump(status_data, f, indent=4)

def preprocess_data(
    df, model_type="rf", target_mode="default", exclude_horse_info=False
):
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df.dropna(subset=["rank"], inplace=True)

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

    base_exclude_cols = ["prize", "rank", "time", "target", "date"]
    if exclude_horse_info:
        base_exclude_cols.extend(["horse_num", "horse_weight", "weight_change"])

    X = df.drop(columns=base_exclude_cols, errors="ignore")
    y = df["target"]

    # Fill missing values for all types
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            if X[col].isnull().all():
                X[col] = X[col].fillna(0)
            else:
                X[col] = X[col].fillna(X[col].median())
        else:
            if X[col].isnull().all():
                X[col] = X[col].fillna("missing")
            else:
                X[col] = X[col].fillna(X[col].mode()[0])

    if model_type == "rf":
        high_cardinality_features = ["horse_name", "jockey"]
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
        return X, y, target_maps

    elif model_type == "lgbm":
        categorical_features = [col for col in X.columns if X[col].dtype == "object"]
        # カテゴリカル特徴量のカテゴリを保持
        categorical_features_with_categories = {}
        for col in categorical_features:
            # NaNを'missing'として扱う
            X[col] = X[col].fillna('missing')
            # カテゴリを明示的に指定してastype('category')
            # 訓練データに存在する全てのユニークなカテゴリを取得
            all_categories = sorted(X[col].unique().tolist())
            X[col] = pd.Categorical(X[col], categories=all_categories)
            categorical_features_with_categories[col] = all_categories

        print(f"Categorical features with categories for LGBM: {categorical_features_with_categories}")
        return X, y, categorical_features_with_categories

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
        y_cnn = to_categorical(df["target"], num_classes=len(df["target"].unique()))

        # Calculate class weights
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(df["target"]),
            y=df["target"]
        )
        class_weight_dict = dict(zip(np.unique(df["target"]), class_weights))

        return [X_seq, X_flat], y_cnn, X_flat.columns.tolist(), imputation_values, class_weight_dict

    else:
        raise ValueError("Invalid model_type specified.")

def build_multi_input_cnn_model(sequence_input_shape, flat_input_shape, num_classes):
    seq_input = Input(shape=sequence_input_shape, name="sequence_input")
    x1 = Conv1D(filters=32, kernel_size=2, activation='relu')(seq_input)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling1D(pool_size=2)(x1)
    x1 = Dropout(0.25)(x1)
    x1 = Conv1D(filters=64, kernel_size=2, activation='relu')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Flatten()(x1)

    flat_input = Input(shape=flat_input_shape, name="flat_input")
    x2 = Dense(32, activation="relu")(flat_input)
    x2 = BatchNormalization()(x2)

    concatenated = concatenate([x1, x2])
    final_output = Dense(128, activation="relu")(concatenated)
    final_output = BatchNormalization()(final_output)
    final_output = Dropout(0.25)(final_output)
    final_output = Dense(num_classes, activation="softmax")(final_output)

    model = Model(inputs=[seq_input, flat_input], outputs=final_output)
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model

def train_model(model_type, X, y, target_mode, horse_info="included", hf_api=None, hf_token=None, hf_model_repo_id=None, **kwargs):
    print(
        f"\n--- Training {model_type.upper()} ({target_mode}, horse_info: {horse_info}) ---"
    )
    if (isinstance(X, list) and (X[0].size == 0 or X[1].size == 0)) or (
        not isinstance(X, list) and X.empty
    ):
        print(f"No data to train the {model_type} model. Exiting training.")
        return

    model_path = get_model_path(model_type, target_mode, horse_info)
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

        model = build_multi_input_cnn_model(
            (X[0].shape[1], X[0].shape[2]), (X[1].shape[1],), y.shape[1]
        )
        model.summary()

        model.fit(
            X_train,
            y_train,
            epochs=10,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=1,
            class_weight=kwargs.get("class_weight_dict")
        )
        model.save(model_path)
        with open(model_path + ".flat_features.json", "w") as f:
            json.dump(kwargs["flat_features_columns"], f)
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
                    path_in_repo=os.path.basename(model_path) + ".imputation_values.json",
                    repo_id=hf_model_repo_id,
                    repo_type="model",
                    token=hf_token,
                )
                print(f"Uploaded CNN model and metadata to Hugging Face: {hf_model_repo_id}")
            except Exception as e:
                print(f"Error uploading CNN model to Hugging Face: {e}")

    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        if model_type == "rf":
            model = RandomForestClassifier(
                n_estimators=100, random_state=42, class_weight="balanced"
            )
            model.fit(X_train, y_train)
            joblib.dump(model, model_path)
            with open(model_path + ".feature_columns.json", "w") as f:
                json.dump(X_train.columns.tolist(), f)
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
                        path_in_repo=os.path.basename(model_path) + ".feature_columns.json",
                        repo_id=hf_model_repo_id,
                        repo_type="model",
                        token=hf_token,
                    )
                    if "target_maps" in kwargs:
                        hf_api.upload_file(
                            path_or_fileobj=model_path + ".target_maps.json",
                            path_in_repo=os.path.basename(model_path) + ".target_maps.json",
                            repo_id=hf_model_repo_id,
                            repo_type="model",
                            token=hf_token,
                        )
                    print(f"Uploaded RF model and metadata to Hugging Face: {hf_model_repo_id}")
                except Exception as e:
                    print(f"Error uploading RF model to Hugging Face: {e}")

        elif model_type == "lgbm":
            model = lgb.LGBMClassifier(random_state=42, class_weight="balanced")
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
            # カテゴリカル特徴量のカテゴリ情報を保存
            if "categorical_features_with_categories" in kwargs:
                save_lgbm_categorical_features(model_path, kwargs["categorical_features_with_categories"])

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
                        path_in_repo=os.path.basename(model_path) + ".feature_columns.json",
                        repo_id=hf_model_repo_id,
                        repo_type="model",
                        token=hf_token,
                    )
                    if "categorical_features_with_categories" in kwargs:
                        hf_api.upload_file(
                            path_or_fileobj=model_path + ".categorical_features.json",
                            path_in_repo=os.path.basename(model_path) + ".categorical_features.json",
                            repo_id=hf_model_repo_id,
                            repo_type="model",
                            token=hf_token,
                        )
                    print(f"Uploaded LGBM model and metadata to Hugging Face: {hf_model_repo_id}")
                except Exception as e:
                    print(f"Error uploading LGBM model to Hugging Face: {e}")

    print(f"{model_type.upper()} Model saved to {model_path}")
