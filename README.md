# Regime-Aware Retirement Portfolio with HMM + CVaR
This repository implements a **dynamic retirement portfolio strategy** that combines **Hidden Markov Models (HMM)** for regime detection with **Conditional Value-at-Risk (CVaR) optimization** for risk-aware asset allocation.  
The goal is to **ensure sustainable cash flows during retirement withdrawals** by adapting portfolio weights to market regimes and incorporating income-oriented assets such as covered call ETFs.  

---

## 📂 Repository Contents
- `portfolio_pipeline.py` : Backtesting pipeline with HMM-based regime classification,  
  CVaR optimization, withdrawal simulation, transaction cost, and ETF expense ratio handling
- `monthly_dividend.py` : Yahoo Finance data fetcher, covered call ETF backtester,  
  additional utility functions
- `analysis/` : Jupyter notebooks for data preprocessing, backtesting runs, and result analysis
- `RegimeAware_Retirement_Portfolio.pdf` : Final project report

---

## 📊 Key Features
- Regime detection: SPY & VIX daily log-returns classified into Bull / Neutral / Bear via HMM
- Risk-aware optimization: Mean-CVaR with ℓ2 regularization to balance return, tail risk, and diversification
- Cash flow simulation: Monthly withdrawals, dividend income, transaction costs, ETF expense ratios included
- Rebalancing analysis: Compare performance under monthly, quarterly, semiannual, and annual schedules
