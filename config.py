"""config.py — GP Indicator Evolution configuration."""

import os
from datetime import datetime

# ── Data ──────────────────────────────────────────────────────────────────────
HF_DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
HF_DATA_FILE = "master_data.parquet"
HF_OUTPUT_REPO = "P2SAMAPA/p2-etf-gp-indicator-evolution-results"
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# ── Universes ─────────────────────────────────────────────────────────────────
FI_COMMODITIES_TICKERS = ["TLT", "GLD", "SLV", "LQD", "HYG", "VNQ"]
EQUITY_SECTORS_TICKERS = [
    "SPY",
    "QQQ",
    "XLK",
    "XLF",
    "XLE",
    "XLV",
    "XLI",
    "XLY",
    "XLP",
    "XLU",
]
COMBINED_TICKERS = FI_COMMODITIES_TICKERS + EQUITY_SECTORS_TICKERS

UNIVERSES = {
    "FI_COMMODITIES": FI_COMMODITIES_TICKERS,
    "EQUITY_SECTORS": EQUITY_SECTORS_TICKERS,
    "COMBINED": COMBINED_TICKERS,
}

# ── Macro features ────────────────────────────────────────────────────────────
MACRO_COLS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M"]

# ── Feature engineering ───────────────────────────────────────────────────────
LOOKBACK_DAYS = 5  # lag features: ret_lag_1 … ret_lag_5

# ── GP parameters ─────────────────────────────────────────────────────────────
POPULATION_SIZE = 300  # was 100 — bigger population → better exploration
GENERATIONS = 50  # was 20 — more generations → better convergence
HALL_OF_FAME_SIZE = 5  # keep top 5 individuals
CROSSOVER_PROB = 0.7
MUTATION_PROB = 0.2
TOURNAMENT_SIZE = 5  # was 3 — stronger selection pressure
INIT_DEPTH_MIN = 2
INIT_DEPTH_MAX = 4  # was 3
MAX_DEPTH = 6  # was 4 — allow richer formulas
TRAIN_RATIO = 0.7  # was 0.8 — more test data for reliable fitness

# ── Transaction costs ─────────────────────────────────────────────────────────
TRANSACTION_COST = 0.001

# ── Consensus windows — 3-year rolling, 2008–2025 ────────────────────────────
CONSENSUS_WINDOWS = [
    (2008, 2011),
    (2009, 2012),
    (2010, 2013),
    (2011, 2014),
    (2012, 2015),
    (2013, 2016),
    (2014, 2017),
    (2015, 2018),
    (2016, 2019),
    (2017, 2020),
    (2018, 2021),
    (2019, 2022),
    (2020, 2023),
    (2021, 2024),
    (2022, 2025),
]

MIN_OBSERVATIONS = 200
TODAY = datetime.now().strftime("%Y-%m-%d")
