import pandas as pd
import numpy as np
import config
import data_manager
from gp_evolver import run_gp
from collections import Counter

def build_data(returns_series, macro_df, risk_free, lookback):
    macro_aligned = macro_df.reindex(returns_series.index, method='ffill')
    rf_aligned = risk_free.reindex(returns_series.index, method='ffill')
    data = pd.DataFrame(index=returns_series.index)
    data['ret'] = returns_series
    for col in macro_aligned.columns:
        data[col] = macro_aligned[col]
    for lag in range(1, lookback+1):
        data[f'ret_lag_{lag}'] = data['ret'].shift(lag)
    data['target'] = data['ret'].shift(-1)
    data['rf'] = rf_aligned
    data.dropna(inplace=True)
    features = [f'ret_lag_{i}' for i in range(1, lookback+1)] + list(macro_aligned.columns)
    X = data[features].values
    y = data['target'].values
    rf = data['rf'].values
    return X, y, rf, features

def run_window(ticker, start, end, returns_df, macro_df, risk_free):
    mask = (returns_df.index >= f"{start}-01-01") & (returns_df.index <= f"{end}-12-31")
    ret_win = returns_df[ticker].loc[mask].dropna()
    if len(ret_win) < config.MIN_OBSERVATIONS:
        return None
    X, y, rf, fnames = build_data(ret_win, macro_df, risk_free, config.LOOKBACK_DAYS)
    if len(X) < 100:
        return None
    split = int(len(X) * config.TRAIN_RATIO)
    X_train, y_train, rf_train = X[:split], y[:split], rf[:split]
    X_test, y_test, rf_test = X[split:], y[split:], rf[split:]
    best_ind, best_str = run_gp(ticker, X_train, y_train, X_test, y_test, rf_test, fnames)
    return best_str

def main():
    if not config.HF_TOKEN:
        print("HF_TOKEN not set")
        return
    df = data_manager.load_master_data()
    macro = data_manager.prepare_macro_features(df)
    # TBILL_3M is annual % (e.g., 4.5). Convert to daily simple return.
    risk_free = macro['TBILL_3M'] / 100.0
    risk_free = (1 + risk_free) ** (1/252) - 1
    returns = data_manager.prepare_returns_matrix(df, config.UNIVERSES["EQUITY_SECTORS"])
    macro, returns = data_manager.align_macro_returns(returns, macro)
    
    results = {}
    for ticker in returns.columns:
        print(f"\n{ticker}")
        win_formulas = {}
        for start, end in config.CONSENSUS_WINDOWS:
            print(f"  {start}-{end}")
            f = run_window(ticker, start, end, returns, macro, risk_free)
            if f:
                win_formulas[f"{start}-{end}"] = f
        if win_formulas:
            cnt = Counter(win_formulas.values())
            top, votes = cnt.most_common(1)[0]
            results[ticker] = {
                "formula": top,
                "consensus_votes": votes,
                "total_windows": len(win_formulas),
                "percentage": votes/len(win_formulas)*100,
                "all_windows": win_formulas
            }
    # push to HF (omitted for brevity – use your push_results.py)
    print("\n=== Done ===")
    print(results)

if __name__ == "__main__":
    main()
