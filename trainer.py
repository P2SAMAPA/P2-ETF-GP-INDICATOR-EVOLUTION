"""
Main trainer for GP Indicator Evolution – corrected.
"""

import pandas as pd
import numpy as np
import config
import data_manager
from gp_evolver import run_gp_for_window
import push_results
from collections import Counter

def build_features_and_targets(returns_series, macro_df, risk_free_series, lookback=20):
    """
    Build X (features), y (next day return), rf (next day risk‑free).
    Returns X, y, rf, feature_names
    """
    macro_aligned = macro_df.reindex(returns_series.index, method='ffill')
    rf_aligned = risk_free_series.reindex(returns_series.index, method='ffill')
    data = pd.DataFrame(index=returns_series.index)
    data['ret'] = returns_series
    for col in macro_aligned.columns:
        data[col] = macro_aligned[col]
    for lag in range(1, lookback+1):
        data[f'ret_lag_{lag}'] = data['ret'].shift(lag)
    data['target'] = data['ret'].shift(-1)   # next day return
    data['rf'] = rf_aligned
    data.dropna(inplace=True)
    feature_cols = [f'ret_lag_{i}' for i in range(1, lookback+1)] + list(macro_aligned.columns)
    X = data[feature_cols].values
    y = data['target'].values
    rf = data['rf'].values
    return X, y, rf, feature_cols

def walk_forward_pairs(X, y, rf, train_days=252, test_days=63, num_folds=5):
    """
    Create overlapping train/test splits. Returns list of folds.
    Each fold: (X_train, y_train, X_test, y_test, rf_test)
    """
    total = len(X)
    if total < train_days + test_days:
        return []
    step = max(1, (total - train_days - test_days) // (num_folds - 1))
    pairs = []
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
        # Standardise features on training set only, then transform test
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        pairs.append((X_train_scaled, y_train, X_test_scaled, y_test, rf_test))
    return pairs

def run_consensus_window(ticker, start_year, end_year, returns_df, macro_df, risk_free_series):
    """
    Run GP for a single consensus window (e.g., 2008-2012).
    Returns best formula string or None.
    """
    mask = (returns_df.index >= pd.Timestamp(f"{start_year}-01-01")) & (returns_df.index <= pd.Timestamp(f"{end_year}-12-31"))
    returns_win = returns_df[ticker].loc[mask].dropna()
    if len(returns_win) < config.MIN_OBSERVATIONS:
        print(f"    Skipped: insufficient returns ({len(returns_win)})")
        return None
    macro_win = macro_df.reindex(returns_win.index, method='ffill')
    rf_win = risk_free_series.reindex(returns_win.index, method='ffill')
    X, y, rf, feature_names = build_features_and_targets(returns_win, macro_win, rf_win, config.LOOKBACK_DAYS)
    if len(X) < config.TRAIN_DAYS + config.TEST_DAYS:
        print(f"    Skipped: insufficient samples after features ({len(X)})")
        return None
    pairs = walk_forward_pairs(X, y, rf, config.TRAIN_DAYS, config.TEST_DAYS, config.NUM_FOLDS)
    if len(pairs) < 2:
        print(f"    Skipped: only {len(pairs)} walk‑forward folds")
        return None
    print(f"    Running GP on {len(pairs)} folds...")
    best_ind, best_str = run_gp_for_window(ticker, pairs, feature_names)
    return best_str

def main():
    import os
    if not config.HF_TOKEN:
        print("HF_TOKEN not set")
        return
    df_master = data_manager.load_master_data()
    macro = data_manager.prepare_macro_features(df_master)
    # Risk‑free rate: TBILL_3M is annual % (e.g., 4.5 for 4.5%)
    # Convert to decimal and then to daily simple return
    risk_free_series = macro['TBILL_3M'] / 100.0   # now decimal (0.045)
    risk_free_series = (1 + risk_free_series) ** (1/252) - 1  # daily simple return
    
    all_results = {}
    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== {universe_name} ===")
        returns_all = data_manager.prepare_returns_matrix(df_master, tickers)
        if len(returns_all) < config.MIN_OBSERVATIONS:
            print("  Skipped: not enough returns")
            continue
        macro_aligned, returns_aligned = data_manager.align_macro_returns(returns_all, macro)
        ticker_consensus = {}
        for ticker in tickers:
            if ticker not in returns_aligned.columns:
                continue
            print(f"  Processing {ticker}...")
            window_formulas = {}
            for start, end in config.CONSENSUS_WINDOWS:
                print(f"    Window {start}-{end}")
                formula = run_consensus_window(ticker, start, end, returns_aligned, macro_aligned, risk_free_series)
                if formula:
                    window_formulas[f"{start}-{end}"] = formula
            if window_formulas:
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
    
    output = {"run_date": config.TODAY, "universes": all_results}
    push_results.push_daily_result(output)
    print("\n=== Evolution Complete ===")

if __name__ == "__main__":
    main()
