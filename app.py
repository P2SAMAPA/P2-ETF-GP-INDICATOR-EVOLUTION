"""app.py — GP Indicator Evolution Dashboard."""

from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from huggingface_hub import HfFileSystem

import config
from us_calendar import next_trading_day

st.set_page_config(
    page_title="GP Indicator Evolution · P2Quant",
    layout="wide",
    page_icon="🧬",
)

# ── Styling ───────────────────────────────────────────────────────────────────
SIGNAL_COLOURS = {"STRONG": "#27AE60", "MODERATE": "#F39C12", "WEAK": "#E74C3C"}
RANK_COLOURS = ["#FFD700", "#C0C0C0", "#CD7F32"]  # gold / silver / bronze


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Loading latest GP results…")
def load_latest() -> dict | None:
    try:
        fs = HfFileSystem(token=config.HF_TOKEN)
        files = fs.ls(f"datasets/{config.HF_OUTPUT_REPO}", detail=False)
        json_files = sorted([f for f in files if f.endswith(".json")])
        if not json_files:
            return None

        # Load the most recent file per universe (parallel jobs write separate files)
        universe_files: dict[str, str] = {}
        for f in json_files:
            name = f.split("/")[-1]  # basename
            # filename: gp_indicator_YYYY-MM-DD_universe-slug.json
            parts = name.replace(".json", "").split("_")
            universe_slug = parts[-1] if len(parts) >= 3 else "all"
            universe_files[universe_slug] = f  # keeps latest (files are sorted)

        merged_universes: dict = {}
        run_date = "unknown"
        for slug, filepath in universe_files.items():
            with fs.open(filepath, "r") as fp:
                data = json.load(fp)
            run_date = data.get("run_date", run_date)
            merged_universes.update(data.get("universes", {}))

        return {"run_date": run_date, "universes": merged_universes}
    except Exception as e:
        st.error(f"Failed to load results: {e}")
        return None


def signal_badge(strength: str) -> str:
    colour = SIGNAL_COLOURS.get(strength, "#888")
    return f'<span style="background:{colour};color:white;padding:3px 10px;border-radius:12px;font-weight:bold;font-size:13px">{strength}</span>'


# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("# 🧬 GP Indicator Evolution")
st.caption(
    "Genetic Programming evolves Sortino-optimised trading signals across 15 rolling "
    "3-year windows · Ranks ETFs by mean out-of-sample Sortino ratio"
)

data = load_latest()
if not data:
    st.warning("⚠️ No results found. Run `trainer.py` first.")
    st.stop()

run_date = data.get("run_date", "unknown")
next_trade = next_trading_day()

col_h1, col_h2, col_h3 = st.columns(3)
col_h1.metric("Run Date", run_date)
col_h2.metric("Next Trading Day", str(next_trade))
col_h3.metric("Universes", len(data.get("universes", {})))

st.divider()

universes = data.get("universes", {})
if not universes:
    st.warning("No universe data found in results.")
    st.stop()

# ── Universe tabs ─────────────────────────────────────────────────────────────
tab_labels = list(universes.keys())
tabs = st.tabs(
    [
        f"{'🏦' if 'FI' in t else '📊' if 'EQUITY' in t else '🌐'} {t}"
        for t in tab_labels
    ]
)

for tab, universe_name in zip(tabs, tab_labels):
    with tab:
        uni = universes[universe_name]
        tickers_data = uni.get("tickers", {})
        rankings = uni.get("rankings", [])

        if not tickers_data or not rankings:
            st.info("No data for this universe.")
            continue

        # Sort by rank
        sorted_tickers = sorted(
            tickers_data.items(), key=lambda x: x[1].get("rank", 99)
        )
        top = sorted_tickers[0]
        top_ticker, top_info = top

        # ── Hero card ─────────────────────────────────────────────────────────
        st.markdown(f"## 🚀 Top Pick for {next_trade}")

        hero_col, metric_col = st.columns([3, 2])
        with hero_col:
            strength = top_info.get("signal_strength", "WEAK")
            st.markdown(
                f"### {top_ticker} &nbsp; {signal_badge(strength)}",
                unsafe_allow_html=True,
            )
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Mean Sortino", f"{top_info.get('mean_sortino', 0):.3f}")
            m2.metric("Win Rate", f"{top_info.get('win_rate_pct', 0):.0f}%")
            m3.metric(
                "Positive Windows",
                f"{top_info.get('positive_windows', 0)}/{top_info.get('total_windows', 0)}",
            )
            m4.metric("Best Sortino", f"{top_info.get('best_sortino', 0):.3f}")

            with st.expander("📜 Best Evolved Formula"):
                st.code(top_info.get("best_formula", "N/A"), language="python")
                st.caption(f"From window: {top_info.get('best_window', 'N/A')}")

        with metric_col:
            # Window-by-window Sortino sparkline for top ticker
            window_scores = top_info.get("window_scores", {})
            if window_scores:
                ws_df = pd.DataFrame(
                    list(window_scores.items()), columns=["Window", "Sortino"]
                ).sort_values("Window")
                fig_spark = go.Figure()
                fig_spark.add_trace(
                    go.Bar(
                        x=ws_df["Window"],
                        y=ws_df["Sortino"],
                        marker_color=[
                            "#27AE60" if v > 0 else "#E74C3C" for v in ws_df["Sortino"]
                        ],
                        name="Sortino",
                    )
                )
                fig_spark.add_hline(y=0, line_dash="dot", line_color="gray")
                fig_spark.update_layout(
                    title=f"{top_ticker} — Sortino by Window",
                    height=220,
                    margin=dict(t=35, b=40, l=40, r=10),
                    xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
                    yaxis_title="Sortino",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                )
                st.plotly_chart(fig_spark, use_container_width=True)

        st.divider()

        # ── Full rankings table with bar chart ────────────────────────────────
        st.subheader(f"📊 Full Rankings — {universe_name}")

        ranks_rows = []
        for ticker, info in sorted_tickers:
            ranks_rows.append(
                {
                    "Rank": info.get("rank", "-"),
                    "Ticker": ticker,
                    "Signal": info.get("signal_strength", "WEAK"),
                    "Mean Sortino": info.get("mean_sortino", 0),
                    "Win Rate %": info.get("win_rate_pct", 0),
                    "Positive Windows": f"{info.get('positive_windows',0)}/{info.get('total_windows',0)}",
                    "Best Sortino": info.get("best_sortino", 0),
                    "Worst Sortino": info.get("worst_sortino", 0),
                    "Best Window": info.get("best_window", ""),
                }
            )
        ranks_df = pd.DataFrame(ranks_rows)

        # Horizontal bar chart of mean Sortino
        chart_col, table_col = st.columns([1, 1])
        with chart_col:
            fig_bar = go.Figure()
            colours = [
                SIGNAL_COLOURS.get(info.get("signal_strength", "WEAK"), "#888")
                for _, info in sorted_tickers
            ]
            tickers_sorted = [t for t, _ in sorted_tickers]
            sortinos = [info.get("mean_sortino", 0) for _, info in sorted_tickers]
            fig_bar.add_trace(
                go.Bar(
                    y=tickers_sorted,
                    x=sortinos,
                    orientation="h",
                    marker_color=colours,
                    text=[f"{s:.3f}" for s in sortinos],
                    textposition="outside",
                )
            )
            fig_bar.add_vline(x=0, line_dash="dot", line_color="gray")
            fig_bar.update_layout(
                title="Mean Sortino Ratio (all windows)",
                height=max(300, len(tickers_sorted) * 35),
                margin=dict(t=40, b=20, l=60, r=60),
                xaxis_title="Mean Sortino",
                yaxis=dict(autorange="reversed"),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with table_col:
            st.dataframe(
                ranks_df.style.format(
                    {
                        "Mean Sortino": "{:.3f}",
                        "Win Rate %": "{:.1f}%",
                        "Best Sortino": "{:.3f}",
                        "Worst Sortino": "{:.3f}",
                    }
                ),
                use_container_width=True,
                height=max(300, len(ranks_df) * 35 + 40),
            )

        # ── Heatmap: Sortino per ticker per window ────────────────────────────
        st.subheader("🌡️ Sortino Heatmap — All Tickers × All Windows")
        heat_data = {}
        all_windows = sorted(
            {w for _, info in sorted_tickers for w in info.get("window_scores", {})}
        )
        for ticker, info in sorted_tickers:
            ws = info.get("window_scores", {})
            heat_data[ticker] = [ws.get(w, float("nan")) for w in all_windows]

        heat_df = pd.DataFrame(heat_data, index=all_windows).T

        fig_heat = go.Figure(
            data=go.Heatmap(
                z=heat_df.values.tolist(),
                x=all_windows,
                y=heat_df.index.tolist(),
                colorscale="RdYlGn",
                zmid=0,
                colorbar=dict(title="Sortino"),
                hoverongaps=False,
            )
        )
        fig_heat.update_layout(
            height=max(300, len(heat_df) * 30 + 80),
            margin=dict(t=20, b=60, l=60, r=20),
            xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # ── Per-ticker detail expander ─────────────────────────────────────────
        st.subheader("🔬 Per-Ticker Detail")
        for ticker, info in sorted_tickers:
            rank = info.get("rank", "-")
            mean_s = info.get("mean_sortino", 0)
            strength = info.get("signal_strength", "WEAK")
            colour = SIGNAL_COLOURS.get(strength, "#888")
            with st.expander(
                f"#{rank}  {ticker}  —  Mean Sortino: {mean_s:.3f}  |  {strength}"
            ):
                d1, d2, d3 = st.columns(3)
                d1.metric("Mean Sortino", f"{mean_s:.3f}")
                d2.metric("Win Rate", f"{info.get('win_rate_pct', 0):.0f}%")
                d3.metric(
                    "Windows",
                    f"{info.get('positive_windows',0)}+ / {info.get('total_windows',0)}",
                )

                ws = info.get("window_scores", {})
                if ws:
                    ws_df2 = pd.DataFrame(
                        list(ws.items()), columns=["Window", "Sortino"]
                    ).sort_values("Window")
                    fig2 = go.Figure()
                    fig2.add_trace(
                        go.Bar(
                            x=ws_df2["Window"],
                            y=ws_df2["Sortino"],
                            marker_color=[
                                "#27AE60" if v > 0 else "#E74C3C"
                                for v in ws_df2["Sortino"]
                            ],
                        )
                    )
                    fig2.add_hline(y=0, line_dash="dot", line_color="gray")
                    fig2.update_layout(
                        height=200,
                        margin=dict(t=10, b=40, l=40, r=10),
                        xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        showlegend=False,
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                best_formula = info.get("best_formula", "")
                if best_formula:
                    st.code(best_formula, language="python")
                    st.caption(f"Best window: {info.get('best_window', '')}")

st.divider()
st.caption(
    f"P2Quant GP Engine · Run: {run_date} · "
    "Data: P2SAMAPA/fi-etf-macro-signal-master-data · "
    "Results: P2SAMAPA/p2-etf-gp-indicator-evolution-results"
)
