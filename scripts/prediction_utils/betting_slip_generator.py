import pandas as pd
from itertools import combinations, permutations

def calculate_min_payout(bet_type, horse_names, df_all_horses_odds, bet_amount=100):
    """
    Estimates the minimum payout for a given betting slip to avoid torikami.
    This is a heuristic, especially for combination bets, as actual combination odds are not available.
    """
    if not horse_names:
        return 0

    # Get odds for the horses in the slip
    slip_odds = []
    for name in horse_names:
        odds_row = df_all_horses_odds[df_all_horses_odds['horse_name'] == name]
        if not odds_row.empty:
            slip_odds.append(odds_row['odds'].iloc[0])
        else:
            # If odds not found, assume 0 or a very low value to avoid betting
            return 0

    if not slip_odds:
        return 0

    if bet_type in ["win", "place"]:
        # For single horse bets, payout is odds * bet_amount
        return slip_odds[0] * bet_amount
    elif bet_type in ["wide", "umaren", "quinella"]:
        # For 2-horse combination bets, a very rough estimate.
        # Actual wide/umaren/quinella odds are usually higher than individual odds.
        # To avoid torikami, we need payout > bet_amount.
        # Let's assume a minimum payout factor based on the lowest odds in the combination.
        # This is a very simplified heuristic.
        min_individual_odds = min(slip_odds)
        # If the lowest individual odds is already less than 1.0, it's likely torikami.
        if min_individual_odds < 1.0:
            return 0
        # A very rough estimate: assume payout is at least the lowest individual odds * some factor
        # For 2-horse combinations, a factor of 1.5-2.0 might be reasonable for avoiding torikami
        # if the individual odds are decent. Let's use a conservative factor.
        return min_individual_odds * 1.2 * bet_amount # Heuristic factor
    elif bet_type in ["umatan", "trio", "trifecta"]:
        # For 3-horse combination bets, even rougher estimate.
        min_individual_odds = min(slip_odds)
        if min_individual_odds < 1.0:
            return 0
        return min_individual_odds * 1.0 * bet_amount # Heuristic factor

    return 0 # Default for unknown bet types

def generate_betting_slips(final_predicted_ranks, value_bets_df, df_all_horses_odds, df_final_for_prediction, bet_type="win", bet_amount=100):
    """
    Generates betting slips based on prediction and value analysis, avoiding torikami.

    Args:
        final_predicted_ranks (list): List of predicted rank categories for each horse.
        value_bets_df (pd.DataFrame): DataFrame with value bet analysis results.
        df_all_horses_odds (pd.DataFrame): DataFrame containing all horses' names and odds in the race.
        df_final_for_prediction (pd.DataFrame): DataFrame containing all race data including horse_class.
        bet_type (str): The type of bet to generate slips for.
        bet_amount (int): The assumed bet amount per slip (e.g., 100 yen).

    Returns:
        list: A list of dictionaries, where each dictionary represents a betting slip.
    """
    betting_slips = []

    if value_bets_df.empty:
        return betting_slips

    # Add predicted ranks and horse_class to value_bets_df for sorting and filtering
    value_bets_df_extended = value_bets_df.copy()
    value_bets_df_extended['predicted_rank'] = final_predicted_ranks
    value_bets_df_extended = value_bets_df_extended.merge(df_final_for_prediction[['horse_name', 'horse_class']], on='horse_name', how='left')

    # Sort by expected value (descending) and then by predicted rank (ascending)
    sorted_value_horses = value_bets_df_extended.sort_values(by=['expected_value', 'predicted_rank'], ascending=[False, True])

    # Filter for actual value bets
    value_horses = sorted_value_horses[sorted_value_horses['is_value_bet']]

    if value_horses.empty:
        return betting_slips

    # Separate horses into core (favorites/contenders) and filler (long_shots/others)
    # Prioritize horses with higher model_prob within their class for selection
    core_horses = value_horses[value_horses['horse_class'].isin(['favorite', 'contender'])].sort_values(by='model_prob', ascending=False)
    filler_horses = value_horses[value_horses['horse_class'].isin(['long_shot', 'other'])].sort_values(by='model_prob', ascending=False)

    # Limit the number of horses to consider for combinations to avoid explosion
    # Adjust limits based on typical betting strategies
    core_horses_limited = core_horses.head(3) # Top 3 core horses
    filler_horses_limited = filler_horses.head(5) # Top 5 filler horses

    if bet_type == "win":
        if not value_horses.empty:
            best_bet = value_horses.iloc[0]
            payout = calculate_min_payout("win", [best_bet['horse_name']], df_all_horses_odds, bet_amount)
            if payout > bet_amount:
                betting_slips.append({
                    "type": "win",
                    "horse_name": best_bet['horse_name'],
                    "odds": best_bet['odds'],
                    "expected_value": best_bet['expected_value'],
                    "estimated_payout": payout
                })

    elif bet_type == "place":
        for _, row in value_horses.iterrows():
            payout = calculate_min_payout("place", [row['horse_name']], df_all_horses_odds, bet_amount)
            if payout > bet_amount:
                betting_slips.append({
                    "type": "place",
                    "horse_name": row['horse_name'],
                    "odds": row['odds'],
                    "expected_value": row['expected_value'],
                    "estimated_payout": payout
                })

    elif bet_type == "wide":
        # Wide: Core-Core, Core-Filler, Filler-Filler combinations
        # Prioritize Core-Core, then Core-Filler, then Filler-Filler
        
        # Core-Core combinations
        if len(core_horses_limited) >= 2:
            for combo in combinations(core_horses_limited.iterrows(), 2):
                horse_names = [horse['horse_name'] for _, horse in combo]
                payout = calculate_min_payout("wide", horse_names, df_all_horses_odds, bet_amount)
                if payout > bet_amount:
                    betting_slips.append({
                        "type": "wide",
                        "horse_names": horse_names,
                        "estimated_payout": payout
                    })
        
        # Core-Filler combinations
        if len(core_horses_limited) >= 1 and len(filler_horses_limited) >= 1:
            for core_horse_idx, core_horse in core_horses_limited.iterrows():
                for filler_horse_idx, filler_horse in filler_horses_limited.iterrows():
                    horse_names = [core_horse['horse_name'], filler_horse['horse_name']]
                    payout = calculate_min_payout("wide", horse_names, df_all_horses_odds, bet_amount)
                    if payout > bet_amount:
                        betting_slips.append({
                            "type": "wide",
                            "horse_names": horse_names,
                            "estimated_payout": payout
                        })

        # Filler-Filler combinations (only if few core horses or many filler horses)
        if len(core_horses_limited) < 2 and len(filler_horses_limited) >= 2:
            for combo in combinations(filler_horses_limited.iterrows(), 2):
                horse_names = [horse['horse_name'] for _, horse in combo]
                payout = calculate_min_payout("wide", horse_names, df_all_horses_odds, bet_amount)
                if payout > bet_amount:
                    betting_slips.append({
                        "type": "wide",
                        "horse_names": horse_names,
                        "estimated_payout": payout
                    })

    elif bet_type == "umaren":
        # Umaren: Core-Core, Core-Filler, Filler-Filler combinations
        # Similar logic to wide, but for umaren
        
        # Core-Core combinations
        if len(core_horses_limited) >= 2:
            for combo in combinations(core_horses_limited.iterrows(), 2):
                horse_names = [horse['horse_name'] for _, horse in combo]
                payout = calculate_min_payout("umaren", horse_names, df_all_horses_odds, bet_amount)
                if payout > bet_amount:
                    betting_slips.append({
                        "type": "umaren",
                        "horse_names": horse_names,
                        "estimated_payout": payout
                    })
        
        # Core-Filler combinations
        if len(core_horses_limited) >= 1 and len(filler_horses_limited) >= 1:
            for core_horse_idx, core_horse in core_horses_limited.iterrows():
                for filler_horse_idx, filler_horse in filler_horses_limited.iterrows():
                    horse_names = [core_horse['horse_name'], filler_horse['horse_name']]
                    payout = calculate_min_payout("umaren", horse_names, df_all_horses_odds, bet_amount)
                    if payout > bet_amount:
                        betting_slips.append({
                            "type": "umaren",
                            "horse_names": horse_names,
                            "estimated_payout": payout
                        })

        # Filler-Filler combinations
        if len(core_horses_limited) < 2 and len(filler_horses_limited) >= 2:
            for combo in combinations(filler_horses_limited.iterrows(), 2):
                horse_names = [horse['horse_name'] for _, horse in combo]
                payout = calculate_min_payout("umaren", horse_names, df_all_horses_odds, bet_amount)
                if payout > bet_amount:
                    betting_slips.append({
                        "type": "umaren",
                        "horse_names": horse_names,
                        "estimated_payout": payout
                    })

    elif bet_type == "umatan":
        # Umatan: Core-Core, Core-Filler, Filler-Filler permutations
        # Similar logic to umaren, but for permutations
        
        # Core-Core permutations
        if len(core_horses_limited) >= 2:
            for combo in permutations(core_horses_limited.iterrows(), 2):
                horse_names = [horse['horse_name'] for _, horse in combo]
                payout = calculate_min_payout("umatan", horse_names, df_all_horses_odds, bet_amount)
                if payout > bet_amount:
                    betting_slips.append({
                        "type": "umatan",
                        "horse_names": horse_names,
                        "estimated_payout": payout
                    })
        
        # Core-Filler permutations
        if len(core_horses_limited) >= 1 and len(filler_horses_limited) >= 1:
            for core_horse_idx, core_horse in core_horses_limited.iterrows():
                for filler_horse_idx, filler_horse in filler_horses_limited.iterrows():
                    horse_names = [core_horse['horse_name'], filler_horse['horse_name']]
                    payout = calculate_min_payout("umatan", horse_names, df_all_horses_odds, bet_amount)
                    if payout > bet_amount:
                        betting_slips.append({
                            "type": "umatan",
                            "horse_names": horse_names,
                            "estimated_payout": payout
                        })

        # Filler-Filler permutations
        if len(core_horses_limited) < 2 and len(filler_horses_limited) >= 2:
            for combo in permutations(filler_horses_limited.iterrows(), 2):
                horse_names = [horse['horse_name'] for _, horse in combo]
                payout = calculate_min_payout("umatan", horse_names, df_all_horses_odds, bet_amount)
                if payout > bet_amount:
                    betting_slips.append({
                        "type": "umatan",
                        "horse_names": horse_names,
                        "estimated_payout": payout
                    })

    elif bet_type == "quinella":
        # Quinella: Core-Core, Core-Filler, Filler-Filler combinations
        # Similar logic to wide, but for quinella
        
        # Core-Core combinations
        if len(core_horses_limited) >= 2:
            for combo in combinations(core_horses_limited.iterrows(), 2):
                horse_names = [horse['horse_name'] for _, horse in combo]
                payout = calculate_min_payout("quinella", horse_names, df_all_horses_odds, bet_amount)
                if payout > bet_amount:
                    betting_slips.append({
                        "type": "quinella",
                        "horse_names": horse_names,
                        "estimated_payout": payout
                    })
        
        # Core-Filler combinations
        if len(core_horses_limited) >= 1 and len(filler_horses_limited) >= 1:
            for core_horse_idx, core_horse in core_horses_limited.iterrows():
                for filler_horse_idx, filler_horse in filler_horses_limited.iterrows():
                    horse_names = [core_horse['horse_name'], filler_horse['horse_name']]
                    payout = calculate_min_payout("quinella", horse_names, df_all_horses_odds, bet_amount)
                    if payout > bet_amount:
                        betting_slips.append({
                            "type": "quinella",
                            "horse_names": horse_names,
                            "estimated_payout": payout
                        })

        # Filler-Filler combinations
        if len(core_horses_limited) < 2 and len(filler_horses_limited) >= 2:
            for combo in combinations(filler_horses_limited.iterrows(), 2):
                horse_names = [horse['horse_name'] for _, horse in combo]
                payout = calculate_min_payout("quinella", horse_names, df_all_horses_odds, bet_amount)
                if payout > bet_amount:
                    betting_slips.append({
                        "type": "quinella",
                        "horse_names": horse_names,
                        "estimated_payout": payout
                    })

    elif bet_type == "trio":
        # Trio: Core-Core-Core, Core-Core-Filler, Core-Filler-Filler, Filler-Filler-Filler combinations
        
        # Core-Core-Core combinations
        if len(core_horses_limited) >= 3:
            for combo in combinations(core_horses_limited.iterrows(), 3):
                horse_names = [horse['horse_name'] for _, horse in combo]
                payout = calculate_min_payout("trio", horse_names, df_all_horses_odds, bet_amount)
                if payout > bet_amount:
                    betting_slips.append({
                        "type": "trio",
                        "horse_names": horse_names,
                        "estimated_payout": payout
                    })
        
        # Core-Core-Filler combinations
        if len(core_horses_limited) >= 2 and len(filler_horses_limited) >= 1:
            for core_combo in combinations(core_horses_limited.iterrows(), 2):
                for filler_horse_idx, filler_horse in filler_horses_limited.iterrows():
                    horse_names = [core_horse['horse_name'] for _, core_horse in core_combo] + [filler_horse['horse_name']]
                    payout = calculate_min_payout("trio", horse_names, df_all_horses_odds, bet_amount)
                    if payout > bet_amount:
                        betting_slips.append({
                            "type": "trio",
                            "horse_names": horse_names,
                            "estimated_payout": payout
                        })

        # Core-Filler-Filler combinations
        if len(core_horses_limited) >= 1 and len(filler_horses_limited) >= 2:
            for core_horse_idx, core_horse in core_horses_limited.iterrows():
                for filler_combo in combinations(filler_horses_limited.iterrows(), 2):
                    horse_names = [core_horse['horse_name']] + [filler_horse['horse_name'] for _, filler_horse in filler_combo]
                    payout = calculate_min_payout("trio", horse_names, df_all_horses_odds, bet_amount)
                    if payout > bet_amount:
                        betting_slips.append({
                            "type": "trio",
                            "horse_names": horse_names,
                            "estimated_payout": payout
                        })

        # Filler-Filler-Filler combinations (only if few core horses)
        if len(core_horses_limited) < 3 and len(filler_horses_limited) >= 3:
            for combo in combinations(filler_horses_limited.iterrows(), 3):
                horse_names = [horse['horse_name'] for _, horse in combo]
                payout = calculate_min_payout("trio", horse_names, df_all_horses_odds, bet_amount)
                if payout > bet_amount:
                    betting_slips.append({
                        "type": "trio",
                        "horse_names": horse_names,
                        "estimated_payout": payout
                    })

    elif bet_type == "trifecta":
        # Trifecta: Core-Core-Core, Core-Core-Filler, Core-Filler-Filler, Filler-Filler-Filler permutations
        
        # Core-Core-Core permutations
        if len(core_horses_limited) >= 3:
            for combo in permutations(core_horses_limited.iterrows(), 3):
                horse_names = [horse['horse_name'] for _, horse in combo]
                payout = calculate_min_payout("trifecta", horse_names, df_all_horses_odds, bet_amount)
                if payout > bet_amount:
                    betting_slips.append({
                        "type": "trifecta",
                        "horse_names": horse_names,
                        "estimated_payout": payout
                    })
        
        # Core-Core-Filler permutations
        if len(core_horses_limited) >= 2 and len(filler_horses_limited) >= 1:
            for core_combo in permutations(core_horses_limited.iterrows(), 2):
                for filler_horse_idx, filler_horse in filler_horses_limited.iterrows():
                    horse_names = [core_horse['horse_name'] for _, core_horse in core_combo] + [filler_horse['horse_name']]
                    payout = calculate_min_payout("trifecta", horse_names, df_all_horses_odds, bet_amount)
                    if payout > bet_amount:
                        betting_slips.append({
                            "type": "trifecta",
                            "horse_names": horse_names,
                            "estimated_payout": payout
                        })

        # Core-Filler-Filler permutations
        if len(core_horses_limited) >= 1 and len(filler_horses_limited) >= 2:
            for core_horse_idx, core_horse in core_horses_limited.iterrows():
                for filler_combo in permutations(filler_horses_limited.iterrows(), 2):
                    horse_names = [core_horse['horse_name']] + [filler_horse['horse_name'] for _, filler_horse in filler_combo]
                    payout = calculate_min_payout("trifecta", horse_names, df_all_horses_odds, bet_amount)
                    if payout > bet_amount:
                        betting_slips.append({
                            "type": "trifecta",
                            "horse_names": horse_names,
                            "estimated_payout": payout
                        })

        # Filler-Filler-Filler permutations (only if few core horses)
        if len(core_horses_limited) < 3 and len(filler_horses_limited) >= 3:
            for combo in permutations(filler_horses_limited.iterrows(), 3):
                horse_names = [horse['horse_name'] for _, horse in combo]
                payout = calculate_min_payout("trifecta", horse_names, df_all_horses_odds, bet_amount)
                if payout > bet_amount:
                    betting_slips.append({
                        "type": "trifecta",
                        "horse_names": horse_names,
                        "estimated_payout": payout
                    })

    return betting_slips

def suggest_bet_types(value_bets_df, num_horses_in_race):
    """
    Suggests appropriate bet types based on value betting analysis and race characteristics.

    Args:
        value_bets_df (pd.DataFrame): DataFrame with value bet analysis results.
        num_horses_in_race (int): Total number of horses in the race.

    Returns:
        list: A list of suggested bet types (strings).
    """
    suggestions = []
    num_value_bets = value_bets_df['is_value_bet'].sum()

    if num_value_bets >= 1:
        suggestions.append("win")
        suggestions.append("place")

    if num_value_bets >= 2:
        suggestions.append("wide")
        suggestions.append("umaren")
        suggestions.append("umatan")

    if num_value_bets >= 3:
        suggestions.append("quinella")
        suggestions.append("trio")
        suggestions.append("trifecta")

    # Further refine suggestions based on number of horses in race
    if num_horses_in_race < 8 and "trifecta" in suggestions:
        # For small fields, trifecta might be too risky if not enough strong value bets
        suggestions.remove("trifecta")
        if "trio" in suggestions:
            suggestions.remove("trio")

    return suggestions