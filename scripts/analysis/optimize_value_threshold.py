#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import joblib
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts.prediction_utils.model_loader import load_all_models
from scripts.prediction_utils.data_preprocessor import preprocess_data_for_prediction
from scripts.prediction_utils.predictor import predict_with_all_models
from scripts.prediction_utils.constants import PLACE_NUM # PLACE_NUMをインポート

# race_idを生成する関数を定義
def generate_race_id(row):
    """Generate race_id from a row of the dataset."""
    try:
        year = str(row["year"])
        place_code = PLACE_NUM[row["place"]]
        kai = str(row["kai"]).zfill(2)
        day = str(row["day"]).zfill(2)
        race_num = str(row["race_num"]).zfill(2)
        return f"{year}{place_code}{kai}{day}{race_num}"
    except KeyError:
        return None


def calculate_roi(df_results):
    """Calculates the Return on Investment (ROI) for a given set of bets."""
    total_bet = len(df_results)
    if total_bet == 0:
        return 0.0
    
    #単勝的中したベットのオッズの合計を計算
    total_return = df_results[df_results['rank'] == 1]['odds'].sum()
    
    roi = (total_return / total_bet) * 100
    return roi

def main():
    print("Loading historical data...")
    try:
        df_all = pd.read_parquet("data/index_dataset.parquet")
        df_odds = pd.read_parquet("data/odds_dataset.parquet")
        
        # df_allにrace_idを生成して追加
        df_all['race_id'] = df_all.apply(generate_race_id, axis=1)
        df_all.dropna(subset=['race_id'], inplace=True)

        # race_idとhorse_nameをキーに結合
        df_all = pd.merge(df_all, df_odds, on=['race_id', 'horse_name'], how='left')

    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure both index_dataset.parquet and odds_dataset.parquet exist.")
        return

    # オッズとランクが有効なデータに絞り込む
    df_all.dropna(subset=['odds', 'rank'], inplace=True)
    df_all = df_all[df_all['odds'] > 0]
    df_all['rank'] = pd.to_numeric(df_all['rank'], errors='coerce')
    df_all.dropna(subset=['rank'], inplace=True)

    print("Loading models...")
    # target_mode='default'のRFモデルのみをロード（バリューベット判定で使われているため）
    all_models = load_all_models()
    target_mode = 'default'
    model_key = 'lgbm_included' # RFからLGBMに変更
    
    if target_mode not in all_models or model_key not in all_models[target_mode]:
        print(f"Required model {model_key} for target_mode {target_mode} not found. Exiting.")
        return

    model_info = all_models[target_mode][model_key]

    print("Preprocessing data for prediction...")
    # preprocess_data_for_predictionに必要な引数を準備
    lgbm_target_maps = model_info.get("target_maps") # target_mapsはLGBMでは使われないが、引数として渡す
    expected_columns = model_info.get("expected_columns")
    categorical_features_with_categories = model_info.get("categorical_features_with_categories")
    
    preprocessed_data = preprocess_data_for_prediction(
        df_all.copy(),
        model_type="lgbm", # model_typeをlgbmに変更
        target_maps=lgbm_target_maps,
        expected_columns=expected_columns,
        categorical_features_with_categories=categorical_features_with_categories
    )

    print("Making predictions on historical data...")
    # 予測確率を計算（キャリブレーションも適用される）
    predictor_input = {model_key: preprocessed_data}
    predictions = predict_with_all_models(all_models, predictor_input, target_mode)
    model_predictions = predictions[model_info['model_path']]

    # 1着になる確率（キャリブレーション済み）
    win_probabilities = model_predictions[:, 0]

    df_all['model_prob_win'] = win_probabilities
    df_all['implied_prob'] = 1 / df_all['odds']
    df_all['expected_value'] = df_all['model_prob_win'] / df_all['implied_prob']

    print("Optimizing value threshold...")
    thresholds = np.arange(1.0, 2.5, 0.1)
    results = []

    for threshold in thresholds:
        value_bets_df = df_all[df_all['expected_value'] > threshold]
        roi = calculate_roi(value_bets_df)
        num_bets = len(value_bets_df)
        results.append({
            'threshold': threshold,
            'roi': roi,
            'num_bets': num_bets
        })
        print(f"Threshold: {threshold:.1f}, Bets: {num_bets}, ROI: {roi:.2f}%")

    # 最適な閾値を決定
    if not results:
        print("No results to determine the optimal threshold.")
        return
        
    optimal_result = max(results, key=lambda x: x['roi'])

    print("\n--- Optimal Value Bet Threshold ---")
    print(f"Optimal Threshold: {optimal_result['threshold']:.1f}")
    print(f"Number of Bets at Optimal Threshold: {optimal_result['num_bets']}")
    print(f"ROI at Optimal Threshold: {optimal_result['roi']:.2f}%")

if __name__ == "__main__":
    main()
