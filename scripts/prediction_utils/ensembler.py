import numpy as np
import json
import os

def ensemble_predictions(all_model_predictions, target_mode):
    if not all_model_predictions:
        return []

    # Get the number of horses from any prediction
    num_horses = len(list(all_model_predictions.values())[0])

    # Initialize weighted probabilities
    if target_mode == "default":
        # 3 classes: 0 (1st), 1 (2-3rd), 2 (Others)
        weighted_probabilities = np.zeros((num_horses, 3))
    elif target_mode == "top3":
        # 2 classes: 0 (1-3rd), 1 (Others)
        weighted_probabilities = np.zeros((num_horses, 2))
    else:
        raise ValueError(f"Unknown target_mode: {target_mode}")

    total_weight = 0
    for model_path, preds in all_model_predictions.items():
        if preds is not None and len(preds) == num_horses:
            # Load metrics for weighting
            metrics_path = model_path + ".metrics.json"
            weight = 1.0  # Default weight if metrics not found or F1 score is missing
            if os.path.exists(metrics_path):
                try:
                    with open(metrics_path, 'r') as f:
                        metrics = json.load(f)
                    if 'f1_score' in metrics and metrics['f1_score'] is not None:
                        weight = metrics['f1_score']
                        print(f"Using F1 score {weight:.4f} as weight for model: {os.path.basename(model_path)}")
                    else:
                        print(f"Warning: F1 score not found or is None in {metrics_path}. Using default weight 1.0.")
                except Exception as e:
                    print(f"Error loading metrics from {metrics_path}: {e}. Using default weight 1.0.")
            else:
                print(f"Warning: Metrics file not found for {model_path}. Using default weight 1.0.")

            weighted_probabilities += preds * weight
            total_weight += weight
        else:
            print(f"Warning: Skipping invalid predictions from {model_path}")

    if total_weight == 0:
        print("Error: No valid model predictions to ensemble or total weight is zero.")
        return ["Error"] * num_horses

    # Normalize the weighted probabilities
    avg_probabilities = weighted_probabilities / total_weight

    final_predictions = [""] * num_horses

    if target_mode == "default":
        # Class indices: 0 for 1st, 1 for 2-3rd, 2 for Others
        horse_probs = []
        for i in range(num_horses):
            horse_probs.append((i, avg_probabilities[i][0], avg_probabilities[i][1])) # (index, prob_1st, prob_2-3rd)
        
        # Sort by prob_1st in descending order
        horse_probs_sorted_by_1st = sorted(horse_probs, key=lambda x: x[1], reverse=True)
        
        # Assign 1st place
        if len(horse_probs_sorted_by_1st) > 0:
            first_place_horse_idx = horse_probs_sorted_by_1st[0][0]
            final_predictions[first_place_horse_idx] = "1st"
            
            # Remove the 1st place horse from consideration for 2-3rd
            remaining_horses = [hp for hp in horse_probs_sorted_by_1st if hp[0] != first_place_horse_idx]
            
            # Sort remaining by prob_2-3rd in descending order
            remaining_horses_sorted_by_2_3rd = sorted(remaining_horses, key=lambda x: x[2], reverse=True)
            
            # Assign 2-3rd places (up to 2 horses)
            assigned_2_3rd_count = 0
            for hp in remaining_horses_sorted_by_2_3rd:
                if assigned_2_3rd_count < 2:
                    final_predictions[hp[0]] = "2-3rd"
                    assigned_2_3rd_count += 1
                else:
                    break
        
        # Fill remaining as Others
        for i in range(num_horses):
            if final_predictions[i] == "":
                final_predictions[i] = "Others"
        
        return final_predictions
        
    elif target_mode == "top3":
        # Class indices: 0 for 1-3rd, 1 for Others
        horse_probs = []
        for i in range(num_horses):
            horse_probs.append((i, avg_probabilities[i][0])) # (index, prob_1-3rd)
        
        horse_probs_sorted_by_1_3rd = sorted(horse_probs, key=lambda x: x[1], reverse=True)
        
        assigned_1_3rd_count = 0
        for hp in horse_probs_sorted_by_1_3rd:
            if assigned_1_3rd_count < 3:
                final_predictions[hp[0]] = "1-3rd"
                assigned_1_3rd_count += 1
            else:
                break
        
        # Fill remaining as Others
        for i in range(num_horses):
            if final_predictions[i] == "":
                final_predictions[i] = "Others"
        
        return final_predictions
    else:
        raise ValueError(f"Unknown target_mode: {target_mode}")
