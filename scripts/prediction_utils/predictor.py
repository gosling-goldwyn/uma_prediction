import lightgbm as lgb
import tensorflow as tf

import numpy as np

def predict_with_all_models(all_models, preprocessed_data, target_mode):
    predictions = {}
    for model_key, data in preprocessed_data.items():
        model_info = all_models[target_mode].get(model_key)
        if model_info:
            model = model_info["model"]
            model_path = model_info["model_path"]
            
            # Get raw probabilities
            if "cnn" in model_key:
                raw_predictions = model.predict(data)
            else:
                raw_predictions = model.predict_proba(data)

            # Calibrate probabilities if calibrators exist
            if "calibrators" in model_info and model_info["calibrators"]:
                calibrated_predictions = np.zeros_like(raw_predictions)
                calibrators = model_info["calibrators"]
                for i in range(raw_predictions.shape[1]):
                    calibrated_predictions[:, i] = calibrators[i].predict(raw_predictions[:, i])
                
                # Normalize to sum to 1
                sum_of_probs = np.sum(calibrated_predictions, axis=1, keepdims=True)
                # Avoid division by zero
                sum_of_probs[sum_of_probs == 0] = 1
                normalized_predictions = calibrated_predictions / sum_of_probs
                predictions[model_path] = normalized_predictions
            else:
                predictions[model_path] = raw_predictions

    return predictions
