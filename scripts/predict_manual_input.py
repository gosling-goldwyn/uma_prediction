import pandas as pd
import numpy as np
import tensorflow as tf
import os

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

# --- Constants ---
from scripts.model_training.train import get_cnn_model_path

# --- Constants ---

# CNNモデルの入力特徴量
SEQUENCE_FEATURES = [
    "1st_speed_idx", "2nd_speed_idx", "3rd_speed_idx", "4th_speed_idx", "5th_speed_idx",
    "1st_lead_idx", "2nd_lead_idx", "3rd_lead_idx", "4th_lead_idx", "5th_lead_idx",
    "1st_pace_idx", "2nd_pace_idx", "3rd_pace_idx", "4th_pace_idx", "5th_pace_idx",
    "1st_rising_idx", "2nd_rising_idx", "3rd_rising_idx", "4th_rising_idx", "5th_rising_idx",
]

# --- Model Prediction Functions (from train.py) ---
def preprocess_data_for_prediction(df, model_type="cnn"):
    """推論用にデータを前処理します。"""
    if model_type == "cnn":
        for col in SEQUENCE_FEATURES:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].fillna(df[col].median() if not df[col].isnull().all() else 0.0)
            else:
                print(f"Warning: CNN feature '{col}' not found in data. Filling with 0.")
                df[col] = 0.0

        sequence_length = 5
        num_features_per_step = len(SEQUENCE_FEATURES) // sequence_length

        if len(df) == 0 or num_features_per_step == 0:
            return np.array([]).reshape(0, sequence_length, 0)

        X_cnn = df[SEQUENCE_FEATURES].values.reshape(len(df), sequence_length, num_features_per_step)
        return X_cnn
    else:
        raise ValueError("Invalid model_type specified. Only 'cnn' is supported for prediction.")

def load_cnn_model(target_mode="default"):
    """学習済みCNNモデルを読み込みます。"""
    cnn_model_path = get_cnn_model_path(target_mode)
    if os.path.exists(cnn_model_path):
        return tf.keras.models.load_model(cnn_model_path)
    else:
        print(f"CNN Model not found at {cnn_model_path}. Please train the model first.")
        return None

def predict_cnn_rank(model, new_data_processed, target_mode="default"):
    """新しいデータに対してCNNモデルで順位を予測します。"""
    if model is None:
        print("CNN Model is not loaded. Cannot make predictions.")
        return None
    if new_data_processed.size == 0:
        print("No data to predict.")
        return []

    predictions = model.predict(new_data_processed)
    predicted_labels = np.argmax(predictions, axis=1)

    if target_mode == "default":
        target_names = {0: "1st", 1: "2-3rd", 2: "Others"}
    elif target_mode == "top3":
        target_names = {0: "1-3rd", 1: "Others"}
    else:
        raise ValueError(f"Unknown target_mode: {target_mode}")

    predicted_categories = [target_names[p] for p in predicted_labels]

    return predicted_categories

# --- Main Logic for Manual Input ---
def get_manual_input_data() -> pd.DataFrame:
    """ユーザーから手動で推論データを入力させます。"""
    print("Please enter the data for prediction.")
    print("For each horse, you will be asked to provide 5 past race data points for speed, lead, pace, and rising indices.")

    num_horses = int(input("Enter the number of horses to predict: "))
    all_horse_data = []

    for i in range(num_horses):
        print(f"\n--- Horse {i+1} ---")
        horse_name = input(f"Enter name for Horse {i+1}: ")
        horse_data = {"horse_name": horse_name}

        for j in range(1, 6): # 1st to 5th past races
            print(f"  -- Past Race {j} --")
            for feature_type in ["speed", "lead", "pace", "rising"]:
                col_name = f"{j}st_{feature_type}_idx"
                while True:
                    try:
                        value = float(input(f"    Enter {feature_type} index for {j}st past race (e.g., 70.5): "))
                        horse_data[col_name] = value
                        break
                    except ValueError:
                        print("Invalid input. Please enter a number.")
        all_horse_data.append(horse_data)

    return pd.DataFrame(all_horse_data)

def predict_with_manual_input():
    """手動入力されたデータで推論を実行します。"""
    print("Starting prediction with manual input...")

    # 1. 学習済みモデルのロード
    cnn_model = load_cnn_model()
    if not cnn_model:
        print("CNN model not loaded. Exiting.")
        return

    # 2. ユーザーからのデータ入力
    df_manual_input = get_manual_input_data()

    if df_manual_input.empty:
        print("No data entered. Exiting.")
        return

    # 3. データの前処理
    print("Preprocessing manual input data for CNN prediction...")
    X_predict = preprocess_data_for_prediction(df_manual_input.copy(), model_type="cnn")

    if X_predict.size == 0:
        print("No data to predict after preprocessing.")
        return

    # 4. 推論の実行
    print("Making predictions...")
    predicted_ranks = predict_cnn_rank(cnn_model, X_predict)

    print("\n--- Prediction Results ---")
    for i, (idx, row) in enumerate(df_manual_input.iterrows()):
        print(f"馬名: {row['horse_name']}, 予測順位カテゴリ: {predicted_ranks[i]}")

if __name__ == "__main__":
    predict_with_manual_input()
