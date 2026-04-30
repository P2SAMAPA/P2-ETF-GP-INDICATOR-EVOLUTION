"""
Main trainer: for each universe, for each ticker, run GP on each consensus window,
then compute consensus formulas across windows.
"""

import pandas as pd
import numpy as np
import config
import data_manager
from gp_evolver import run_gp_for_window
import push_results
import itertools

def build_features_and_targets(returns_series, macro_df, risk_free_series, lookback=20):
    """
    Build X (features), y (next day return), and rf (next day risk‑free).
    """
    # Align macro and risk‑free to returns index
    macro_aligned = macro_df.reindex(returns_series.index, method='ffill')
    rf_aligned = risk_free_series.reindex(returns_series.index, method='ffill')
    data = pd.DataFrame(index=returns_series.index)
    data['ret'] = returns_series
    for col in macro_aligned.columns:
        data[col] = macro_aligned[col]
    for lag in range(1, lookback+1):
        data[f'ret_lag_{lag}'] = data['ret'].shift(lag)
    # Target: next day return (log return)
    data['target'] = data['ret'].shift(-1)
    data['rf'] = rf_aligned
    data.dropna(inplace=True)
    # Feature columns
    feature_cols = [f'ret_lag_{i}' for i in range(1, lookback+1)] + list(macro_aligned.columns)
    X = data[feature_cols].values
    y = data['target'].values
    rf = data['rf'].values
    return X, y, rf

def walk_forward_pairs(X, y, rf, train_days=252, test_days=63, num_folds=5):
    """
    Create overlapping train/test splits for walk‑forward evaluation.
    Returns list of (X_train, y_train, X_test, y_test, rf_test) for each fold.
    """
    total = len(X)
    pairs = []
    # Step size so that num_folds cover the data
    step = max(1, (total - train_days - test_days) // (num_folds - 1))
    for i in range(num_folds):
        train_start = i * step
        train_end = train_start + train_days
        test_end = train_end + test_days
        if test_end > total:
            break
        X_train = X[train_start:train_end]
        y_train = y[train_start:train_end]
        X_test = X[train_end:test_end]
        y_test = y[train_end:test_end]
        rf_test = rf[train_end:test_end]
        pairs.append((X_train, y_train, X_test, y_test, rf_test))
    return pairs

def run_consensus_window(ticker, start_year, end_year, returns_df, macro_df, risk_free_series):
    """
    Run GP for a single consensus window (e.g., 2008-2012).
    Returns best formula string.
    """
    # Filter data to window dates
    mask = (returns_df.index >= pd.Timestamp(f"{start_year}-01-01")) & (returns_df.index <= pd.Timestamp(f"{end_year}-12-31"))
    returns_win = returns_df[ticker].loc[mask].dropna()
    if len(returns_win) < config.MIN_OBSERVATIONS:
        return None
    macro_win = macro_df.reindex(returns_win.index, method='ffill')
    rf_win = risk_free_series.reindex(returns_win.index, method='ffill')
    # Build features
    X, y, rf = build_features_and_targets(returns_win, macro_win, rf_win, lookback=config.LOOKBACK_DAYS)
    # Walk‑forward folds
    pairs = walk_forward_pairs(X, y, rf, config.TRAIN_DAYS, config.TEST_DAYS, config.NUM_FOLDS)
    if len(pairs) < 2:
        return None
    # Run GP
    best_ind, best_str = run_gp_for_window(ticker, returns_win, macro_win, pairs)
    return best_str

def main():
    import os
    if not config.HF_TOKEN:
        print("HF_TOKEN not set")
        return
    df_master = data_manager.load_master_data()
    macro = data_manager.prepare_macro_features(df_master)
    # Risk‑free rate series from TBILL_3M (annual %), convert to daily simple return
    risk_free_series = macro['TBILL_3M'] / 100.0   # annual rate as decimal
    risk_free_series = (1 + risk_free_series) ** (1/252) - 1   # daily simple return
    
    all_results = {}
    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== {universe_name} ===")
        returns_all = data_manager.prepare_returns_matrix(df_master, tickers)
        if len(returns_all) < config.MIN_OBSERVATIONS:
            continue
        macro_aligned, returns_aligned = data_manager.align_macro_returns(returns_all, macro)
        # For each ticker, for each consensus window, evolve formula
        ticker_consensus = {}
        for ticker in tickers:
            if ticker not in returns_aligned.columns:
                continue
            window_formulas = {}
            for start, end in config.CONSENSUS_WINDOWS:
                print(f"  Evolving for {ticker} on window {start}-{end}...")
                formula = run_consensus_window(ticker, start, end, returns_aligned, macro_aligned, risk_free_series)
                if formula:
                    window_formulas[f"{start}-{end}"] = formula
            # Consensus: most frequent formula across windows (exact string match)
            if window_formulas:
                from collections import Counter
                counter = Counter(window_formulas.values())
                top_formula, votes = counter.most_common(1)[0]
                total = len(window_formulas)
                ticker_consensus[ticker] = {
                    "formula": top_formula,
                    "consensus_votes": votes,
                    "total_windows": total,
                    "percentage": votes/total*100,
                    "all_windows": window_formulas
                }
        all_results[universe_name] = ticker_consensus
    
    output = {
        "run_date": config.TODAY,
        "universes": all_results
    }
    push_results.push_daily_result(output)
    print("\n=== Evolution Complete ===")

if __name__ == "__main__":
    main()
