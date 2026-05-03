"""trainer.py — GP Indicator Evolution engine orchestrator."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

import config
import data_manager
import push_results
from gp_evolver import run_gp


def build_window_data(
    ticker: str,
    start_year: int,
    end_year: int,
    returns_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    risk_free: pd.Series,
) -> tuple | None:
    """Slice data to a 3-year window and build feature matrix."""
    mask = (returns_df.index.year >= start_year) & (returns_df.index.year <= end_year)
    ret_win = returns_df[ticker].loc[mask].dropna()

    if len(ret_win) < config.MIN_OBSERVATIONS:
        return None

    macro_aligned = macro_df.reindex(ret_win.index, method="ffill")
    rf_aligned = risk_free.reindex(ret_win.index, method="ffill").fillna(0.0)

    data = pd.DataFrame(index=ret_win.index)
    data["ret"] = ret_win
    for col in macro_aligned.columns:
        data[col] = macro_aligned[col]
    for lag in range(1, config.LOOKBACK_DAYS + 1):
        data[f"ret_lag_{lag}"] = data["ret"].shift(lag)
    data["rf"] = rf_aligned
    data.dropna(inplace=True)

    if len(data) < 100:
        return None

    feature_cols = [f"ret_lag_{i}" for i in range(1, config.LOOKBACK_DAYS + 1)] + list(
        macro_aligned.columns
    )
    X = data[feature_cols].values
    y = data["ret"].values
    rf = data["rf"].values

    split = int(len(X) * config.TRAIN_RATIO)
    if split < 50 or len(X) - split < 50:
        return None

    return (
        X[:split],
        y[:split],
        X[split:],
        y[split:],
        rf[split:],
        feature_cols,
    )


def run_universe(
    universe_name: str,
    tickers: list[str],
    returns_aligned: pd.DataFrame,
    macro_aligned: pd.DataFrame,
    risk_free: pd.Series,
    pop_size: int,
    n_gen: int,
) -> dict:
    """Run GP for all tickers in one universe across all windows."""
    print(f"\n{'='*60}")
    print(f"Universe: {universe_name}  ({len(tickers)} tickers)")
    print(f"  GP: pop={pop_size}  gen={n_gen}")
    print(f"{'='*60}")

    ticker_window_scores: dict[str, dict] = {t: {} for t in returns_aligned.columns}
    ticker_window_formulas: dict[str, dict] = {t: {} for t in returns_aligned.columns}

    for start, end in config.CONSENSUS_WINDOWS:
        window_label = f"{start}-{end}"
        print(f"\n  Window {window_label}")

        for ticker in returns_aligned.columns:
            result = build_window_data(
                ticker, start, end, returns_aligned, macro_aligned, risk_free
            )
            if result is None:
                print(f"    {ticker}: insufficient data — skipping")
                continue

            X_train, y_train, X_test, y_test, rf_test, feature_cols = result
            seed = abs(start * 100 + hash(ticker) % 100)
            score, formula = run_gp(
                ticker,
                X_train,
                y_train,
                X_test,
                y_test,
                rf_test,
                feature_cols,
                seed=seed,
                pop_size=pop_size,
                n_gen=n_gen,
            )
            ticker_window_scores[ticker][window_label] = round(score, 4)
            ticker_window_formulas[ticker][window_label] = formula
            print(f"    {ticker}: Sortino={score:.3f}")

    # Rank tickers by mean Sortino
    universe_results: dict = {}
    rankings: list[tuple[str, float]] = []

    for ticker in returns_aligned.columns:
        scores = ticker_window_scores[ticker]
        if not scores:
            continue
        vals = list(scores.values())
        mean_sortino = float(np.mean(vals))
        positive_windows = sum(1 for s in vals if s > 0)
        win_rate = positive_windows / len(vals)
        best_window = max(scores, key=scores.get)

        universe_results[ticker] = {
            "mean_sortino": round(mean_sortino, 4),
            "win_rate": round(win_rate, 4),
            "win_rate_pct": round(win_rate * 100, 1),
            "positive_windows": positive_windows,
            "total_windows": len(vals),
            "best_window": best_window,
            "best_sortino": round(scores[best_window], 4),
            "worst_sortino": round(min(vals), 4),
            "window_scores": scores,
            "best_formula": ticker_window_formulas[ticker].get(best_window, ""),
            "all_formulas": ticker_window_formulas[ticker],
        }
        rankings.append((ticker, mean_sortino))

    rankings.sort(key=lambda x: x[1], reverse=True)
    for rank, (ticker, score) in enumerate(rankings, 1):
        universe_results[ticker]["rank"] = rank
        signal = "STRONG" if score > 0.5 else "MODERATE" if score > 0.0 else "WEAK"
        universe_results[ticker]["signal_strength"] = signal
        print(
            f"  Rank {rank:2d}  {ticker:6s}  "
            f"MeanSortino={score:.3f}  "
            f"WinRate={universe_results[ticker]['win_rate_pct']:.0f}%  "
            f"Signal={signal}"
        )

    return {"rankings": rankings, "tickers": universe_results}


def main() -> None:
    if not config.HF_TOKEN:
        print("HF_TOKEN not set — aborting.")
        return

    # Which universe to run (from env var set by GitHub Actions matrix)
    target = os.environ.get("GP_UNIVERSE", "all").upper()

    df = data_manager.load_master_data()
    macro = data_manager.prepare_macro_features(df)
    risk_free = macro["TBILL_3M"] / 100.0
    risk_free = (1 + risk_free) ** (1 / 252) - 1

    all_results: dict = {}

    for universe_name, tickers in config.UNIVERSES.items():
        if target != "ALL" and universe_name != target:
            continue

        returns = data_manager.prepare_returns_matrix(df, tickers)
        macro_aligned, returns_aligned = data_manager.align_macro_returns(
            returns, macro
        )

        # COMBINED gets smaller GP to fit in time budget
        if universe_name == "COMBINED":
            pop_size, n_gen = 150, 30
        else:
            pop_size, n_gen = config.POPULATION_SIZE, config.GENERATIONS

        result = run_universe(
            universe_name,
            tickers,
            returns_aligned,
            macro_aligned,
            risk_free,
            pop_size,
            n_gen,
        )
        all_results[universe_name] = result

    output = {"run_date": config.TODAY, "universes": all_results}
    push_results.push_daily_result(output, universe=target)
    print("\n✅ Evolution complete — results pushed to HuggingFace.")


if __name__ == "__main__":
    main()
