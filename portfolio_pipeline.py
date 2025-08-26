# ---------------------------------------------
# portfolio_pipeline.py   ★ All‑Trading‑Day Safe + Withdrawal trace
#                          + HMM full‑period state prediction + first‑month fix
#                          + instrument meta (stock / ETF) with per‑asset
#                            slippage, expense ratio, and global dividend tax
#                          + regime label column (Bull / Bear / Neutral‑k)
# ---------------------------------------------
import numpy as np
import pandas as pd
import warnings
from dataclasses import dataclass, field
from typing import Dict, Union, List
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
from pypfopt import (
    expected_returns,
    risk_models,
    EfficientFrontier,
    EfficientSemivariance,
    EfficientCVaR,
    objective_functions,
)
import random


EPS = 1e-6  # numerical tolerance


random.seed(42)
np.random.seed(42)

# ---------------- instrument meta ----------------
asset_meta: Dict[str, Dict[str, Union[str, float]]] = {
    # ETFs
    "SPY":  {"type": "ETF", "slippage": 0.0001, "expense_ratio": 0.0009},
    "QQQ":  {"type": "ETF", "slippage": 0.0001, "expense_ratio": 0.0020},
    "SCHD": {"type": "ETF", "slippage": 0.0001, "expense_ratio": 0.0006},
    "XYLD": {"type": "ETF", "slippage": 0.0001, "expense_ratio": 0.0060},
    "QYLD": {"type": "ETF", "slippage": 0.0001, "expense_ratio": 0.0060},
    "GOVT": {"type": "ETF", "slippage": 0.0001, "expense_ratio": 0.0005},
    "SHY":  {"type": "ETF", "slippage": 0.0001, "expense_ratio": 0.0015},
    # Stocks (sample basket)
    "MSFT": {"type": "stock", "slippage": 0.0001},
    "AAPL": {"type": "stock", "slippage": 0.0001},
    "GOOGL": {"type": "stock", "slippage": 0.0001},
    "BRK-B": {"type": "stock", "slippage": 0.0002},
    "JPM": {"type": "stock", "slippage": 0.00015},
    "UNH": {"type": "stock", "slippage": 0.0002},
    "KO": {"type": "stock", "slippage": 0.0001},
    "COST": {"type": "stock", "slippage": 0.00025},
    "LYB": {"type": "stock", "slippage": 0.0003},
    "XOM": {"type": "stock", "slippage": 0.0001},
    "NEE": {"type": "stock", "slippage": 0.0002},
}
_DEFAULT_SLIP = 0.0

def _slip(t: str) -> float:
    return asset_meta.get(t, {}).get("slippage", _DEFAULT_SLIP)

def _is_etf(t: str) -> bool:
    return asset_meta.get(t, {}).get("type", "stock").lower() == "etf"

def _expense_ratio(t: str) -> float:
    return asset_meta.get(t, {}).get("expense_ratio", 0.0)

# ---------------- utils ----------------

def month_open(price_df: pd.DataFrame) -> pd.DatetimeIndex:
    s = price_df.index.to_series()
    return pd.DatetimeIndex(s.groupby([s.dt.year, s.dt.month]).head(1).values)

# ---------------- config ----------------
@dataclass
class BacktestConfig:
    initial_cash: float = 200_000
    monthly_withdraw: float = 2_000
    trade_cost: float = 0.001
    slippage: Union[float, Dict[str, float]] = field(default_factory=lambda: _DEFAULT_SLIP)
    dividend_tax: float = 0.0
    beta_cvar: float = 0.95
    target_cvar: float = 0.05
    target_return: float = 0.03
    l2_gamma: float = 1
    window_years: int = 3
    hmm_states: int = 3
    hmm_iter: int = 600
    hmm_tol: float = 1e-2
    freq: int = 252
    start_date: str = "2017-01-01"
    rebalance_freq: str = "monthly"       # monthly / quarterly / semiannual / annual


# ---------------- HMM helpers ----------------

def rank_to_regime(mu_by_state: pd.Series) -> Dict[int, str]:
    """Rank states by mean return → Bull/Bear/Neutral‑k labels."""
    ordered = mu_by_state.sort_values()  # low → high
    n = len(ordered)
    mapping: Dict[int, str] = {}
    for rank, (st, _) in enumerate(ordered.items()):
        if rank == 0:
            mapping[st] = "Bear"
        elif rank == n - 1:
            mapping[st] = "Bull"
        else:
            mapping[st] = "Neutral" if n == 3 else f"Neutral-{rank}"
    return mapping

# ---------------- rolling HMM ----------------

def monthly_state(hmm_df: pd.DataFrame, cal: pd.DatetimeIndex, cfg: BacktestConfig,
                  full_index: pd.DatetimeIndex) -> pd.Series:
    scaler = StandardScaler()
    states = pd.Series(index=full_index, dtype="float")

    for d in cal[cal >= hmm_df.index.min() + pd.DateOffset(years=cfg.window_years)]:
        win_idx = (hmm_df.index >= d - pd.DateOffset(years=cfg.window_years)) & (hmm_df.index < d)
        win = hmm_df.loc[win_idx]
        X = scaler.fit_transform(win.values)
        if X.shape[0] < cfg.hmm_states * 10:
            continue
        model = GaussianHMM(n_components=cfg.hmm_states, covariance_type="diag", min_covar=1e-4,
                            n_iter=cfg.hmm_iter, tol=cfg.hmm_tol, random_state=42).fit(X)
        pred_states = model.predict(X)
        states.loc[win.index] = pred_states
        current_X = scaler.transform(hmm_df.loc[[d]].values)
        states.loc[d] = model.predict(current_X)[0]
    return states.ffill()

# ---------------- optimiser ----------------

def optimise(ret: pd.DataFrame, method: str, cfg: BacktestConfig) -> Dict[str, float]:
    tickers: List[str] = ret.columns.tolist()
    if method == "6040":
        return {t: 0.6/(len(tickers)-1) if t != "SHY" else 0.4 for t in tickers}
    mu = expected_returns.mean_historical_return(ret, returns_data=True, frequency=cfg.freq)
    opts = dict(solver="SCS", verbose=False)
    if method == "MV":
        pr = expected_returns.prices_from_returns(ret)
        cov = risk_models.CovarianceShrinkage(pr).ledoit_wolf()
        ef = EfficientFrontier(None, cov, **opts)
        if method.endswith("L2"):
            ef.add_objective(objective_functions.L2_reg, gamma=cfg.l2_gamma)
        ef.min_volatility(); w = ef.clean_weights()
    elif method.startswith("Semi"):
        es = EfficientSemivariance(mu, ret, **opts)
        if method.endswith("L2"):
            es.add_objective(objective_functions.L2_reg, gamma=cfg.l2_gamma)
        w = es.min_semivariance()
    elif method.startswith("CVaR"):
        ec = EfficientCVaR(mu, ret, beta=cfg.beta_cvar, **opts)
        if method.endswith("L2"):
            ec.add_objective(objective_functions.L2_reg, gamma=cfg.l2_gamma)
        w = ec.efficient_risk(cfg.target_cvar)
    else:
        raise ValueError(f"Unknown optimisation method → {method}")
    return {k: float(v) for k, v in w.items() if abs(v) > EPS}


def _due(freq: str, date: pd.Timestamp) -> bool:
    if freq == "monthly":    return True
    if freq == "quarterly":  return date.month in {1, 4, 7, 10}
    if freq == "semiannual": return date.month in {1, 7}
    if freq == "annual":     return date.month == 1
    raise ValueError("rebalance_freq must be monthly / quarterly / semiannual / annual")


# ---------------- rebalance ----------------

def rebalance(price: pd.Series, tgt_w: Dict[str, float], cash: float,
              pos: Dict[str, float], fee: float):
    tot = cash + sum(pos[t] * price[t] for t in pos)
    target_val = {t: tgt_w.get(t, 0.0) * tot for t in pos}
    for t in pos:  # sell pass
        excess = pos[t]*price[t] - target_val[t]
        if excess > EPS:
            q = excess / price[t]
            slip = _slip(t)
            cash += q*price[t]*(1-fee-slip); pos[t] -= q
    for t in pos:  # buy pass
        need = target_val[t] - pos[t]*price[t]
        if need > EPS and cash > EPS:
            slip = _slip(t)
            q = min(need/price[t]/(1+fee+slip), cash/(price[t]*(1+fee+slip)))
            cash -= q*price[t]*(1+fee+slip); pos[t] += q
    port = cash + sum(pos[t]*price[t] for t in pos)
    w = {f"{t}_w": (pos[t]*price[t]/port if port else 0.0) for t in pos}
    return cash, pos, w

# ---------------- backtest ----------------

def backtest(price_df: pd.DataFrame, adj_price_df: pd.DataFrame, div_df: pd.DataFrame,
             hmm_df: pd.DataFrame, *, use_hmm: bool = True, method: str = "CVaR_L2",
             cfg: BacktestConfig = BacktestConfig()):
    cal = month_open(price_df); cal = cal[cal >= pd.Timestamp(cfg.start_date)]
    state = monthly_state(hmm_df, cal, cfg, price_df.index) if use_hmm else None
    pos = {t: 0.0 for t in price_df.columns}; cash = cfg.initial_cash
    cum_with = cum_div = 0.0; rows = []
    for i, d in enumerate(cal):
        prev = cal[i-1] if i else cal[i]
        div_window = div_df.loc[prev+pd.Timedelta(days=1):d]
        div_by_asset = div_window.mul(pos).sum()  # 시리즈: {ticker: dividend amount}
        net_div_by_asset = div_by_asset * (1 - cfg.dividend_tax)
        net_div = net_div_by_asset.sum()
        cash += net_div
        cum_div += net_div
        monthly_expense = 0.0
        for t, sh in pos.items():
            if sh>EPS and _is_etf(t): monthly_expense += sh*price_df.loc[d,t]*(_expense_ratio(t)/12)
        cash -= monthly_expense
        want = cfg.monthly_withdraw; with_done = 0.0
        if i>0:
            if cash >= want:
                cash -= want; with_done = want
            else:
                shortage = want - cash
                for t in sorted(pos, key=lambda x: pos[x]*price_df.loc[d,x], reverse=True):
                    if shortage <= EPS: break
                    slip = _slip(t)
                    q = min(shortage/(price_df.loc[d,t]*(1-cfg.trade_cost-slip)), pos[t])
                    proceeds = q*price_df.loc[d,t]*(1-cfg.trade_cost-slip)
                    pos[t] -= q; cash += proceeds; shortage -= proceeds
                with_done = want - shortage; cash -= with_done
                if shortage>EPS:
                    cum_with += with_done
                    rows.append([d,0.0,0.0,with_done,net_div,cum_with,cum_with,
                                 int(state.loc[d]) if use_hmm else np.nan,
                                 *pos.values(), *([0.0]*len(pos))]); break
        cum_with += with_done
        win_adj = adj_price_df.loc[d-pd.DateOffset(years=cfg.window_years):d-pd.Timedelta(days=1)]
        ret_full = expected_returns.returns_from_prices(win_adj).dropna()
        ret = ret_full
        if use_hmm:
            st = int(state.loc[d]); ret = ret_full.loc[state.loc[ret_full.index]==st]
        tgt = optimise(ret, method, cfg)
        
        if _due(cfg.rebalance_freq, d):                # ← 추가
            win_adj = adj_price_df.loc[
                d - pd.DateOffset(years=cfg.window_years) : d - pd.Timedelta(days=1)
            ]
            ret_full = expected_returns.returns_from_prices(win_adj).dropna()
            ret = (
                ret_full
                if not use_hmm
                else ret_full.loc[state.loc[ret_full.index] == int(state.loc[d])]
            )
            cash, pos, w_act = rebalance(
                price_df.loc[d],
                optimise(ret, method, cfg),
                cash,
                pos,
                cfg.trade_cost,
            )
        
        port_val = cash + sum(pos[t]*price_df.loc[d,t] for t in pos)
        if i==0: port_val = cfg.initial_cash
        total_val = port_val + cum_with
        rows.append([d, port_val, cash, with_done, net_div, cum_with, total_val,
             int(state.loc[d]) if use_hmm else np.nan,
             *pos.values(), *w_act.values(), *net_div_by_asset.reindex(pos.keys(), fill_value=0).values])


    col_pos = list(pos.keys()); col_w = [f"{t}_w" for t in pos]
    cols = ["Date","port_value","cash","withdraw","div_income","cum_with","total_value","state"] + col_pos + col_w
    cols += [f"{t}_div" for t in col_pos]  # ← 추가
    df = pd.DataFrame(rows, columns=cols).set_index("Date")

    # ---- regime label 추가 ----
    if use_hmm:
        full_ret = expected_returns.returns_from_prices(adj_price_df).dropna()
        mu_state = (
            full_ret.groupby(state.loc[full_ret.index])
            .mean()
            .mean(axis=1)
        )
        regime_map = rank_to_regime(mu_state)
        df["regime"] = df["state"].map(regime_map)
    else:
        df["regime"] = np.nan  # HMM 안 쓴 경우는 NaN 처리

    return df