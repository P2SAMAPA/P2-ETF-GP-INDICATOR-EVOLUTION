# P2-ETF-GP-INDICATOR-EVOLUTION

Genetic Programming for long‑only ETF indicator evolution.  
Evolves interpretable formulas using Sortino fitness, transaction costs, and 17‑window consensus.

## Structure

- `trainer.py` – main evolution loop per universe/ticker/window
- `gp_evolver.py` – DEAP GP implementation
- `data_manager.py` – loads master data, aligns macro
- `app.py` – Streamlit dashboard
- GitHub Actions runs daily

## Outputs

- Hugging Face dataset: `P2SAMAPA/p2-etf-gp-indicator-evolution-results`
- Each ticker gets a consensus formula across 17 overlapping windows
- Dashboard shows formula and consensus strength

## Run locally

```bash
pip install -r requirements.txt
python trainer.py
streamlit run app.py
