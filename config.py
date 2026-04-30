import os
from datetime import datetime

HF_DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
HF_DATA_FILE = "master_data.parquet"
HF_OUTPUT_REPO = "P2SAMAPA/p2-etf-gp-indicator-evolution-results"

FI_COMMODITIES_TICKERS = ["TLT", "GLD", "SLV", "LQD", "HYG", "VNQ"]
EQUITY_SECTORS_TICKERS = ["SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU"]
UNIVERSES = {"EQUITY_SECTORS": EQUITY_SECTORS_TICKERS}   # pick one for testing

MACRO_COLS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M"]
LOOKBACK_DAYS = 5

# Small, fast GP
POPULATION_SIZE = 100
GENERATIONS = 20
HALL_OF_FAME_SIZE = 3
CROSSOVER_PROB = 0.7
MUTATION_PROB = 0.25
TOURNAMENT_SIZE = 3
PARSIMONY_COEFF = 0.0
INIT_DEPTH_MIN = 2
INIT_DEPTH_MAX = 3
MAX_DEPTH = 4

TRAIN_RATIO = 0.8
TRANSACTION_COST = 0.001

# 3‑year windows from 2008 to 2025 (full)
CONSENSUS_WINDOWS = [(2008,2011), (2009,2012), (2010,2013), (2011,2014),
                     (2012,2015), (2013,2016), (2014,2017), (2015,2018),
                     (2016,2019), (2017,2020), (2018,2021), (2019,2022),
                     (2020,2023), (2021,2024), (2022,2025)]

MIN_OBSERVATIONS = 200
TODAY = datetime.now().strftime("%Y-%m-%d")
HF_TOKEN = os.environ.get("HF_TOKEN", None)
