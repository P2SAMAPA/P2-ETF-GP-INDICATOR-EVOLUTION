"""
Streamlit dashboard for GP Indicator Evolution.
Shows top formulas per ticker, consensus statistics, and backtest curves.
"""

import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
from huggingface_hub import HfFileSystem
import config
from us_calendar import next_trading_day

st.set_page_config(page_title="GP Indicator Evolution", layout="wide")
st.title("🧬 P2-ETF-GP-INDICATOR-EVOLUTION")
st.caption("Genetic Programming evolves long‑only trading indicators | Sortino fitness + transaction costs | 17‑window consensus")

@st.cache_data(ttl=3600)
def load_latest():
    fs = HfFileSystem(token=config.HF_TOKEN)
    repo = config.HF_OUTPUT_REPO
    try:
        files = fs.ls(f"datasets/{repo}")
        json_files = []
        for f in files:
            name = f['name'] if isinstance(f, dict) else f
            if name.endswith('.json'):
                json_files.append(name)
        if not json_files:
            return None
        latest = max(json_files)
        with fs.open(latest, "r") as fp:
            return json.load(fp)
    except Exception as e:
        st.error(f"Error loading results: {e}")
        return None

data = load_latest()
if not data:
    st.warning("No results found. Run trainer.py first.")
    st.stop()

st.sidebar.header("ℹ️ Info")
st.sidebar.write(f"**Run date:** {data['run_date']}")
st.sidebar.write(f"**Next trading day:** {next_trading_day()}")
st.sidebar.write("**Fitness:** Annualised Sortino (downside deviation)")
st.sidebar.write("**Transaction cost:** 10 bps")

universes = data['universes']
selected_universe = st.selectbox("Select Universe", list(universes.keys()))
universe_data = universes[selected_universe]

if universe_data:
    st.header(f"📊 {selected_universe}")
    tickers = list(universe_data.keys())
    for ticker in tickers:
        info = universe_data[ticker]
        with st.expander(f"**{ticker}** – consensus reached in {info['percentage']:.1f}% of windows ({info['consensus_votes']}/{info['total_windows']})"):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.code(info['formula'], language='python')
            with col2:
                st.metric("Consensus strength", f"{info['percentage']:.0f}%")
                st.caption(f"Votes: {info['consensus_votes']} / {info['total_windows']} windows")
            if st.checkbox(f"Show all window formulas for {ticker}"):
                for window, formula in info['all_windows'].items():
                    st.markdown(f"**{window}**")
                    st.code(formula, language='python')
else:
    st.info("No evolved formulas for this universe.")

st.divider()
st.caption("Data source: P2SAMAPA/fi-etf-macro-signal-master-data | Results: P2SAMAPA/p2-etf-gp-indicator-evolution-results")
