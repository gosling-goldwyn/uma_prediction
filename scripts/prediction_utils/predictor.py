import lightgbm as lgb
import tensorflow as tf

def predict_with_all_models(models, preprocessed_data, target_mode):
    predictions = {}
    
    # RF models
    for horse_info in ["included", "excluded"]:
        model_key = f"rf_{horse_info}"
        if model_key in models[target_mode]:
            rf_model = models[target_mode][model_key]["model"]
            # RF models predict probabilities for each class
            predictions[model_key] = rf_model.predict_proba(preprocessed_data[model_key])
        else:
            print(f"Warning: RF model {model_key} not available for {target_mode}.")

    # LGBM models
    for horse_info in ["included", "excluded"]:
        model_key = f"lgbm_{horse_info}"
        if model_key in models[target_mode]:
            lgbm_model = models[target_mode][model_key]["model"]
            # LGBM models predict probabilities for each class
            predictions[model_key] = lgbm_model.predict_proba(preprocessed_data[model_key])
        else:
            print(f"Warning: LGBM model {model_key} not available for {target_mode}.")

    # CNN model
    model_key = "cnn_included"
    if model_key in models[target_mode]:
        cnn_model = models[target_mode][model_key]["model"]
        # CNN models predict probabilities for each class
        predictions[model_key] = cnn_model.predict(preprocessed_data[model_key])
    else:
        print(f"Warning: CNN model {model_key} not available for {target_mode}.")

    return predictions
