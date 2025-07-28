import pandas as pd
import numpy as np

def identify_value_bets(df_race_data: pd.DataFrame, model_predictions: np.ndarray, target_mode: str, value_threshold: float = 2.4) -> pd.DataFrame:
    """
    Identifies 'value bets' by comparing model's predicted probabilities with implied probabilities from market odds.

    Args:
        df_race_data: DataFrame containing race data, including 'odds' and 'horse_name'.
        model_predictions: NumPy array of model's predicted probabilities for each horse.
                           For 'default' mode, it's (num_horses, 3) where index 0 is 1st, 1 is 2-3rd, 2 is Others.
                           For 'top3' mode, it's (num_horses, 2) where index 0 is 1-3rd, 1 is Others.
        target_mode: 'default' or 'top3', indicating the prediction target.
        value_threshold: Multiplier for identifying value bets (model_prob / implied_prob > value_threshold).

    Returns:
        DataFrame with horse name, odds, predicted probability, implied probability, expected value, and a boolean
        indicating if it's a value bet. Returns empty DataFrame if odds data is missing or invalid.
    """
    if 'odds' not in df_race_data.columns:
        print("Warning: 'odds' column not found in race data. Cannot perform value betting analysis.")
        return pd.DataFrame()

    df_value_bets = df_race_data[['horse_name', 'odds']].copy()

    # Convert odds to numeric, handling potential errors
    df_value_bets['odds'] = pd.to_numeric(df_value_bets['odds'], errors='coerce')
    df_value_bets.dropna(subset=['odds'], inplace=True)

    if df_value_bets.empty:
        print("Warning: No valid odds data available for value betting analysis.")
        return pd.DataFrame()

    # Calculate implied probability from odds
    df_value_bets['implied_prob'] = 1 / df_value_bets['odds']

    # Get the relevant model probability based on target_mode
    if target_mode == 'default':
        # Use probability of 1st place (index 0)
        df_value_bets['model_prob'] = model_predictions[:, 0]
    elif target_mode == 'top3':
        # Use probability of 1-3rd place (index 0)
        df_value_bets['model_prob'] = model_predictions[:, 0]
    else:
        raise ValueError(f"Unknown target_mode: {target_mode}")

    # Calculate expected value (model_prob / implied_prob)
    df_value_bets['expected_value'] = df_value_bets['model_prob'] / df_value_bets['implied_prob']

    # Identify value bets based on the threshold
    potential_value_bets = df_value_bets[df_value_bets['expected_value'] > value_threshold]

    # Select up to top 3 potential value bets based on expected value
    top_value_bets = potential_value_bets.nlargest(3, 'expected_value')

    # Create the 'is_value_bet' column
    df_value_bets['is_value_bet'] = df_value_bets.index.isin(top_value_bets.index)

    return df_value_bets
