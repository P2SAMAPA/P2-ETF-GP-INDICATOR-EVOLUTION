"""
Streamlit dashboard for GP Indicator Evolution.
Shows actionable trading recommendations for the next US trading day.
"""

import streamlit as st
import pandas as pd
import json
from huggingface_hub import HfFileSystem
import config
from us_calendar import next_trading_day

st.set_page_config(page_title="GP Indicator Evolution", layout="wide", page_icon="📈")
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

next_trade = next_trading_day()
st.info(f"📅 **Next US trading day:** {next_trade}")

universes = data['universes']
universe_names = list(universes.keys())

# Create tabs for each universe
tabs = st.tabs(universe_names)

for tab, universe_name in zip(tabs, universe_names):
    with tab:
        uni_data = universes[universe_name]
        if not uni_data:
            st.info("No evolved formulas for this universe.")
            continue
        
        # Find the ticker with highest consensus votes
        best_ticker = None
        best_votes = -1
        best_info = None
        for ticker, info in uni_data.items():
            if info['consensus_votes'] > best_votes:
                best_votes = info['consensus_votes']
                best_ticker = ticker
                best_info = info
        
        if best_ticker is None:
            st.info("No consensus reached.")
            continue
        
        # Hero card for the recommended ETF
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"## 🚀 Recommended ETF for {next_trade}")
            st.markdown(f"### {best_ticker}")
            st.markdown(f"**Consensus:** {best_info['consensus_votes']} / {best_info['total_windows']} windows ({best_info['percentage']:.1f}%)")
            # Truncate long formula
            formula = best_info['formula']
            if len(formula) > 200:
                formula = formula[:200] + "..."
            with st.expander("📜 Evolved Formula"):
                st.code(formula, language='python')
        with col2:
            strength = "STRONG" if best_info['percentage'] >= 60 else "MODERATE" if best_info['percentage'] >= 30 else "WEAK"
            st.metric("Signal Strength", strength)
            st.caption(f"Based on {best_info['total_windows']} historical windows")
        
        # Other tickers as expandable list
        with st.expander("🔍 See all tickers in this universe"):
            other_df = pd.DataFrame([
                {
                    "Ticker": t,
                    "Consensus %": info['percentage'],
                    "Votes": f"{info['consensus_votes']}/{info['total_windows']}",
                    "Formula": info['formula'][:80] + "..."
                }
                for t, info in uni_data.items()
            ]).sort_values("Consensus %", ascending=False)
            st.dataframe(other_df, use_container_width=True)

st.divider()
st.caption("Data source: P2SAMAPA/fi-etf-macro-signal-master-data | Results: P2SAMAPA/p2-etf-gp-indicator-evolution-results")
