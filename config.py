"""
Configuration for P2-ETF-GP-INDICATOR-EVOLUTION.
"""

import os
from datetime import datetime

# --- Hugging Face ---
HF_DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
HF_DATA_FILE = "master_data.parquet"
HF_OUTPUT_REPO = "P2SAMAPA/p2-etf-gp-indicator-evolution-results"

# --- Universe Definitions (same as before) ---
FI_COMMODITIES_TICKERS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
EQUITY_SECTORS_TICKERS = [
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV",
    "XLI", "XLY", "XLP", "XLU", "GDX", "XME",
    "IWF", "XSD", "XBI", "IWM"
]
ALL_TICKERS = list(set(FI_COMMODITIES_TICKERS + EQUITY_SECTORS_TICKERS))

UNIVERSES = {
    "FI_COMMODITIES": FI_COMMODITIES_TICKERS,
    "EQUITY_SECTORS": EQUITY_SECTORS_TICKERS,
    "COMBINED": ALL_TICKERS
}

# --- Macro Features ---
MACRO_COLS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M"]  # TBILL_3M is risk‑free rate

# --- Feature Engineering ---
LOOKBACK_DAYS = 20          # number of lagged returns for each ticker
FEATURE_COLS = []           # will be built dynamically: lags + macro

# --- GP Parameters ---
POPULATION_SIZE = 200
GENERATIONS = 30
HALL_OF_FAME_SIZE = 10
CROSSOVER_PROB = 0.7
MUTATION_PROB = 0.25
REPRODUCTION_PROB = 0.05
TOURNAMENT_SIZE = 3
PARSIMONY_COEFF = 0.0001     # penalty per node (small)
INIT_DEPTH_MIN = 2
INIT_DEPTH_MAX = 6
MAX_DEPTH = 8

# --- Walk-Forward Parameters ---
TRAIN_DAYS = 252
TEST_DAYS = 63
NUM_FOLDS = 5               # overlapping folds within a window
TRANSACTION_COST = 0.001    # 10 bps

# --- Consensus Windows (17 periods) ---
CONSENSUS_WINDOWS = [
    (2008, 2012), (2009, 2013), (2010, 2014),
    (2011, 2015), (2012, 2016), (2013, 2017),
    (2014, 2018), (2015, 2019), (2016, 2020),
    (2017, 2021), (2018, 2022), (2019, 2023),
    (2020, 2024), (2021, 2025), (2022, 2026),
    (2023, 2027), (2024, 2028)
]

# --- Training ---
MIN_OBSERVATIONS = 252

# --- Date ---
TODAY = datetime.now().strftime("%Y-%m-%d")

# --- Hugging Face Token ---
HF_TOKEN = os.environ.get("HF_TOKEN", None)
