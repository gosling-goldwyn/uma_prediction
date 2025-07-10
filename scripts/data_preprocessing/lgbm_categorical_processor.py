import json
import os

def save_lgbm_categorical_features(model_path, categorical_features_with_categories):
    """
    LGBMモデルのカテゴリカル特徴量とそのカテゴリをJSONファイルとして保存します。
    """
    save_path = model_path + ".categorical_features.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(categorical_features_with_categories, f, indent=4)
    print(f"Saved LGBM categorical features to {save_path}")

def load_lgbm_categorical_features(model_path):
    """
    LGBMモデルのカテゴリカル特徴量とそのカテゴリをJSONファイルからロードします。
    """
    load_path = model_path + ".categorical_features.json"
    if os.path.exists(load_path):
        with open(load_path, "r") as f:
            return json.load(f)
    return None
