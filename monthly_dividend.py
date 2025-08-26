import yfinance as yf
import pandas as pd
import numpy as np
import requests
import datetime
import scipy.stats as si

class YFfetcher:
    @staticmethod
    def get_price_history(code: str, start_date: str, end_date: str) -> pd.DataFrame:
        start_date = datetime.datetime.strptime(start_date + "-13-00", "%Y-%m-%d-%H-%M")
        end_date = datetime.datetime.strptime(end_date + "-13-00", "%Y-%m-%d-%H-%M")

        period1 = int(start_date.timestamp())
        period2 = int(end_date.timestamp())

        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{code}"
        params = {
            "period1": period1,
            "period2": period2,
            "interval": "1d",
            "events": "history",
            "includeAdjustedClose": "true"
        }
        headers = {"User-Agent": "Mozilla/5.0"}

        res = requests.get(url, params=params, headers=headers)
        data = res.json()

        if "chart" not in data or "result" not in data["chart"]:
            raise ValueError("데이터를 찾을 수 없습니다.")

        result = data["chart"]["result"][0]
        timestamps = result["timestamp"]
        indicators = result["indicators"]["quote"][0]
        adjclose = result["indicators"].get("adjclose", [{}])[0].get("adjclose", [None] * len(timestamps))

        df = pd.DataFrame({
            "Date": pd.to_datetime(timestamps, unit='s'),
            "Open": indicators["open"],
            "High": indicators["high"],
            "Low": indicators["low"],
            "Close": indicators["close"],
            "Adj Close": adjclose,
            "Volume": indicators["volume"]
        })

        df.sort_values(by="Date", inplace=True, ignore_index=True)
        return df

    @staticmethod
    def get_dividends(code: str, start_date: str, end_date: str) -> pd.DataFrame:
        start_dt = datetime.datetime.strptime(start_date + "-13-00", "%Y-%m-%d-%H-%M")
        end_dt = datetime.datetime.strptime(end_date + "-13-00", "%Y-%m-%d-%H-%M")
        period1 = int(start_dt.timestamp())
        period2 = int(end_dt.timestamp())

        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{code}"
        params = {
            "period1": period1,
            "period2": period2,
            "interval": "1d",
            "events": "div",
            "includeAdjustedClose": "false"
        }
        headers = {"User-Agent": "Mozilla/5.0"}

        res = requests.get(url, params=params, headers=headers)
        data = res.json()

        try:
            dividends = data["chart"]["result"][0]["events"]["dividends"]
        except (KeyError, IndexError):
            return pd.DataFrame(columns=["Date", "Dividend"])

        records = [
            {"Date": pd.to_datetime(dividends[key]["date"], unit="s"), "Dividend": dividends[key]["amount"]}
            for key in dividends
        ]

        df = pd.DataFrame(records).sort_values(by="Date").reset_index(drop=True)
        return df

class OptionPricer:
    @staticmethod
    def black_scholes_call(S, K, T, r, sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)

class CoveredCallbacktester:
    def __init__(self, close, vix, irx, dividend=None, initial_cash=10000, payout_rate=0.005,
                 dividend_yield=0.015, stock_trading_cost=0.0004, option_trading_cost=0.02,
                 slippage=0.002, T=1/12):
        self.close = close
        self.vix = vix
        self.irx = irx
        self.dividend = dividend
        self.initial_cash = initial_cash
        self.payout_rate = payout_rate
        self.dividend_yield = dividend_yield
        self.stock_trading_cost = stock_trading_cost
        self.option_trading_cost = option_trading_cost
        self.slippage = slippage
        self.T = T

    def run(self):
        cash = self.initial_cash
        holdings = 0
        cumulative_payout = 0

        portfolio_value_list = []
        total_portfolio_value_list = []
        normalized_price_list = []
        payout_list = []
        cumulative_payout_list = []

        for i in range(len(self.close) - 1):
            price, next_price = self.close.iloc[i].item(), self.close.iloc[i+1].item()
            iv = self.vix if isinstance(self.vix, (int, float)) else self.vix.iloc[i].item() / 100
            next_iv = self.vix if isinstance(self.vix, (int, float)) else self.vix.iloc[i+1].item() / 100
            r, next_r = self.irx.iloc[i].item() / 100, self.irx.iloc[i+1].item() / 100
            strike_price, next_strike_price = price * (1 - self.slippage), next_price * (1 - self.slippage)

            if self.dividend is not None:
                next_div = self.dividend.iloc[i+1].item()

            if i == 0:
                holdings = cash / price
                cash = 0
                premium = OptionPricer.black_scholes_call(price, strike_price, self.T, r, iv)
                option_income = premium * holdings
                cash += option_income * (1 - self.option_trading_cost)

                payout_list.append(0)
                cumulative_payout_list.append(cumulative_payout)  # 0
                normalized_price_list.append(self.initial_cash)
                portfolio_value_list.append(self.initial_cash)
                total_portfolio_value_list.append(self.initial_cash)

            if self.dividend is not None:
                cash += holdings * next_div
            else:
                cash += holdings * (self.dividend_yield / 12) * price

            if next_price > strike_price:
                sell_price = strike_price * (1 - self.stock_trading_cost)
                cash += holdings * sell_price
                holdings = cash / next_price
                cash = 0
            else:
                holdings = (holdings * next_price + cash) / next_price
                cash = 0

            premium = OptionPricer.black_scholes_call(next_price, next_strike_price, self.T, next_r, next_iv)
            option_income = premium * holdings
            cash += option_income * (1 - self.option_trading_cost)

            payout = (holdings * next_price + cash) * self.payout_rate
            payout_list.append(payout)
            cumulative_payout += payout
            cumulative_payout_list.append(cumulative_payout)

            if payout > cash:
                cash_needed = payout - cash
                shares_to_sell = cash_needed / (next_price * (1 - self.stock_trading_cost))
                holdings -= shares_to_sell
                cash = 0
            else:
                cash -= payout

            portfolio_value = holdings * next_price + cash
            portfolio_value_list.append(portfolio_value)

            prev_value = portfolio_value_list[-2]
            monthly_return = (portfolio_value + payout) / prev_value
            total_value = total_portfolio_value_list[-1] * monthly_return
            total_portfolio_value_list.append(total_value)

            normalized_price = (next_price / self.close.iloc[0]) * self.initial_cash
            normalized_price_list.append(normalized_price)

        return pd.DataFrame({
            "Normalized_price": normalized_price_list,
            "portfolio_value": portfolio_value_list,
            "pay_out": payout_list,
            "cummulative_payout": cumulative_payout_list,
            "total_portfolio_value": total_portfolio_value_list
        })

#class GBM:
#    @staticmethod
#    def simulate(nsteps=1000, mu=0.0001, sigma=0.02, dt=1, start=1):
#        steps = [(mu - 0.5 * sigma**2) * dt + np.random.randn() * sigma * np.sqrt(dt) for _ in range(nsteps)]
#        y = [start] + list(start * np.exp(np.cumsum(steps)))
#        return y


class GBM:
    @staticmethod
    def simulate(nsteps=1000, mu=0.0001, sigma=0.02, dt=1, start=1, npaths=100):
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt)

        # 난수 생성: shape (npaths, nsteps)
        rand = np.random.randn(npaths, nsteps)

        # 누적합을 위해 로그수익률 계산
        log_returns = drift + diffusion * rand
        log_paths = np.cumsum(log_returns, axis=1)
        log_paths = np.concatenate([np.zeros((npaths, 1)), log_paths], axis=1)  

        # 지수화 후
        paths = start * np.exp(log_paths)
        return paths  # shape: (npaths, nsteps+1)


def evaluate_portfolio(values, risk_free_rate=0.0, periods_per_year=12):
    """
    포트폴리오 가치 시계열 기준으로 각종 지표 계산
    
    Parameters:
        values (array-like): 포트폴리오 가치 시계열
        risk_free_rate (float or array-like): 무위험 수익률 (고정값 또는 시계열)
        periods_per_year (int): 연환산 기준 기간 수 (예: 월간 = 12)

    Returns:
        dict: 수익률, 리스크, 성과 지표들을 포함한 딕셔너리
    """
    values = np.asarray(values)
    returns = values[1:] / values[:-1] - 1

    if isinstance(risk_free_rate, (int, float)):
        rf_series = risk_free_rate
    else:
        rf_series = np.asarray(risk_free_rate)
        if len(rf_series) != len(returns):
            raise ValueError("Length of risk_free_rate must match number of returns.")

    def calculate_sharpe(returns, rf):
        excess = returns - rf if not np.isscalar(rf) else returns - rf
        std = np.std(excess)
        return np.nan if std == 0 else (np.mean(excess) * periods_per_year) / (std * np.sqrt(periods_per_year))

    def calculate_sortino(returns, rf, periods_per_year=12):
        returns = np.asarray(returns)
        rf = np.asarray(rf) if not np.isscalar(rf) else rf

        # 목표 수익률보다 낮은 수익률만 추출
        if np.isscalar(rf):
            downside_returns = returns[returns < rf]
            excess_returns = returns - rf
        else:
            if len(rf) != len(returns):
                raise ValueError("Length of risk_free_rate must match returns.")
            downside_returns = returns[returns < rf]
            excess_returns = returns - rf

        # 하방 리스크 계산
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return np.nan

        downside_std = np.std(downside_returns)
        mean_excess = np.mean(excess_returns)

        return (mean_excess * periods_per_year) / (downside_std * np.sqrt(periods_per_year))

    def calculate_mdd(values):
        peak = np.maximum.accumulate(values)
        drawdowns = (values - peak) / peak
        return np.min(drawdowns)

    total_return = values[-1] / values[0] - 1
    cagr = (1 + total_return) ** (periods_per_year / (len(values) - 1)) - 1
    annual_vol = np.std(returns) * np.sqrt(periods_per_year)

    return {
        "total_return": total_return,
        "cagr": cagr,
        "annualized_volatility": annual_vol,
        "sharpe_ratio": calculate_sharpe(returns, rf_series),
        "sortino_ratio": calculate_sortino(returns, rf_series),
        "max_drawdown": calculate_mdd(values)
    }



#---------------------------------------------------------------------------------------------------------------------
def simulate_withdrawal_etf_only(
    price_df: pd.DataFrame,
    dividend_df: pd.DataFrame | None = None,
    weights: dict[str, float] | None = None,
    *,
    initial_cash: float = 100_000,
    withdrawal: float | None = None,
    stock_trading_cost: float = 0.002,
    currency: str = "usd",
    exchange_rate: pd.Series | None = None,
    add_valuation_cols: bool = True,
):
    """ETF만 보유한 상태에서 고정 인출 시뮬레이션"""
    # ─────────────────────────── 입력 점검 및 환산 ───────────────────────────
    if weights is None:
        weights = {c: 1 / price_df.shape[1] for c in price_df.columns}
    assert abs(sum(weights.values()) - 1) < 1e-6, "Weights must sum to 1."

    if currency.lower() == "krw":
        assert exchange_rate is not None, "exchange_rate 필수"
        price_df = price_df.mul(exchange_rate, axis=0)
        if dividend_df is not None:
            dividend_df = dividend_df.mul(exchange_rate, axis=0)

    # ─────────────────────────── 초기 매수 ───────────────────────────
    fee = stock_trading_cost
    shares = {
        a: (initial_cash * w) / (price_df.iloc[0, price_df.columns.get_loc(a)] * (1 + fee))
        for a, w in weights.items()
    }
    cash = cum_div = cum_wd = 0.0
    depleted_at = None
    rec = []

    # ─────────────────────────── 월별 시뮬레이션 ───────────────────────────
    for i, date in enumerate(price_df.index):
        px = price_df.iloc[i]
        dv = dividend_df.iloc[i] if dividend_df is not None else pd.Series(0, index=px.index)

        # --- 0) 첫 달 기록 ---
        if i == 0:
            port = sum(shares[a] * px[a] for a in shares) + cash
            rec.append((date, port, port, cum_div, cum_wd, cash, *shares.values()))
            continue

        # --- 1) 배당 ---
        div_cash = sum(dv[a] * shares[a] for a in shares)
        cash += div_cash
        cum_div += div_cash

        # --- 2) 고정 인출 ---
        if withdrawal is not None:
            # 2-A) 현금 부족 → 비중별 + 가치순 매도
            if cash + 1e-9 < withdrawal:
                shortage = withdrawal - cash

                # 비중별 1차 매도
                liquid_w = {a: weights[a] for a in shares if shares[a] > 0}
                tot_w = sum(liquid_w.values())
                for a in liquid_w:
                    sell_target = shortage * liquid_w[a] / tot_w
                    max_amt = shares[a] * px[a] * (1 - fee)
                    sell_amt = min(sell_target, max_amt)
                    shares_to_sell = sell_amt / (px[a] * (1 - fee))
                    shares[a] -= shares_to_sell
                    cash += sell_amt
                    shortage -= sell_amt

                # 가치순 2차 매도
                while shortage > 1e-6 and any(sh > 0 for sh in shares.values()):
                    vals = {a: sh * px[a] * (1 - fee) for a, sh in shares.items()}
                    top = max(vals, key=vals.get)
                    sell_amt = min(shortage, vals[top])
                    shares_to_sell = sell_amt / (px[top] * (1 - fee))
                    shares[top] -= shares_to_sell
                    cash += sell_amt
                    shortage -= sell_amt

            # 2-B) **전부 매각해도 부족 → 고갈**  (수정 구간)
            liquidatable = sum(sh * px[a] * (1 - fee) for a, sh in shares.items())
            if cash + liquidatable + 1e-9 < withdrawal:
                # 전량 매각 후 남은 금액까지 인출
                cash += liquidatable
                shares = {a: 0.0 for a in shares}
                cum_wd += cash     # 마지막 현금을 전부 인출
                cash = 0.0

                port_val = 0.0
                tot_val = cum_wd   # ★ 누적 인출액이 총 가치가 되도록 수정

                rec.append((date, port_val, tot_val, cum_div, cum_wd, cash, *shares.values()))
                depleted_at = date
                break

            # 2-C) 정상 인출
            cash -= withdrawal
            cum_wd += withdrawal

        # --- 3) 남은 현금 재투자 ---
        if cash > 0:
            for a in weights:
                buy_amt = cash * weights[a]
                buy_sh = buy_amt / (px[a] * (1 + fee))
                cost = buy_sh * px[a] * (1 + fee)
                if cost < 1e-8:
                    continue
                shares[a] += buy_sh
                cash -= cost
            cash = round(max(cash, 0.0), 10)

        # --- 4) 월말 기록 ---
        port_val = cash + sum(shares[a] * px[a] for a in shares)
        tot_val = port_val + cum_wd
        rec.append((date, port_val, tot_val, cum_div, cum_wd, cash, *shares.values()))

        if depleted_at is not None:
            break

    # ─────────────────────────── 결과 DataFrame ───────────────────────────
    cols = [
        "Date",
        "value_after_withdrawal",
        "total_portfolio_value",
        "cumulative_dividends",
        "cumulative_withdrawals",
        "cash",
        *price_df.columns,
    ]
    result = pd.DataFrame(rec, columns=cols).set_index("Date")

    # --- 선택: 평가금액 열 ---
    if add_valuation_cols:
        val_df = result[price_df.columns].mul(price_df.loc[result.index])
        val_df = val_df.add_prefix("val_")
        result = result.join(val_df)

    return result, depleted_at




def simulate_withdrawal_etf_cash(
    price_df: pd.DataFrame,
    dividend_df: pd.DataFrame,
    weights: dict,
    monthly_rf: pd.Series,
    initial_cash: float = 10000,
    withdrawal: float = 100,
    stock_trading_cost: float = 0.002
):
    assert abs(sum(weights.values()) - 1.0) < 1e-6, "Weights must sum to 1."

    tickers = price_df.columns
    has_cash = "CASH" in weights
    etf_tickers = [t for t in tickers if t != "CASH"]

    # 초기 현금 및 주식 매수
    cash_reserve = initial_cash * weights.get("CASH", 0.0)
    shares = {
        ticker: (initial_cash * weights[ticker]) / (price_df.iloc[0][ticker] * (1 + stock_trading_cost))
        for ticker in etf_tickers
    }

    # 기록용 변수
    cum_dividends = 0.0
    cum_withdrawal = 0.0
    cum_interest = 0.0
    cash_etf = 0.0
    depleted_at = None

    value_after_withdrawal_list = []
    total_values = []
    cumulative_dividends_list = []
    cumulative_withdrawal_list = []
    cumulative_interest_list = []
    cash_etf_list = []
    cash_reserve_list = []
    shares_history = []

    for i in range(len(price_df)):
        date = price_df.index[i]
        prices = price_df.iloc[i]
        divs = dividend_df.iloc[i] if dividend_df is not None else pd.Series(0.0, index=etf_tickers)

        if i == 0:
            port_value = sum(shares[t] * prices[t] for t in etf_tickers) + cash_etf + cash_reserve
            total_value = port_value + cum_withdrawal
            value_after_withdrawal_list.append(port_value)
            total_values.append(total_value)
            cumulative_dividends_list.append(cum_dividends)
            cumulative_withdrawal_list.append(cum_withdrawal)
            cumulative_interest_list.append(cum_interest)
            cash_etf_list.append(cash_etf)
            cash_reserve_list.append(cash_reserve)
            shares_history.append(shares.copy())
            continue

        # 1. 배당 수령
        dividend = sum(divs[t] * shares[t] for t in etf_tickers)
        cash_etf += dividend
        cum_dividends += dividend

        # 2. 이자 수령
        prev_cash = cash_reserve
        cash_reserve *= (1 + monthly_rf[i])
        interest = cash_reserve - prev_cash
        cum_interest += interest

        # 3. 인출
        remaining_withdrawal = withdrawal

        # (1) 배당에서 차감
        if cash_etf >= remaining_withdrawal:
            cash_etf -= remaining_withdrawal
            remaining_withdrawal = 0.0
        else:
            remaining_withdrawal -= cash_etf
            cash_etf = 0.0

        # (2) 현금에서 차감
        if cash_reserve >= remaining_withdrawal:
            cash_reserve -= remaining_withdrawal
            remaining_withdrawal = 0.0
        else:
            remaining_withdrawal -= cash_reserve
            cash_reserve = 0.0

        # (3) ETF 매도
        if remaining_withdrawal > 0:

            available_weights = {t: weights[t] for t in shares if shares[t] > 0}
            total_w = sum(available_weights.values())

            for t in available_weights:
                target_amt = remaining_withdrawal * available_weights[t] / total_w
                max_amt = shares[t] * prices[t] * (1 - stock_trading_cost)
                sell_amt = min(target_amt, max_amt)
                shares_to_sell = sell_amt / (prices[t] * (1 - stock_trading_cost))
                shares[t] -= shares_to_sell
                remaining_withdrawal -= sell_amt
                if remaining_withdrawal <= 1e-6:
                    break

        cum_withdrawal += withdrawal

        # 4. 포트폴리오 평가
        risky_value = sum(shares[t] * prices[t] for t in etf_tickers)
        value_after_withdrawal = risky_value + cash_etf + cash_reserve
        total_value = value_after_withdrawal + cum_withdrawal

        if value_after_withdrawal <= 0 and depleted_at is None:
            depleted_at = date
            value_after_withdrawal = 0

        value_after_withdrawal_list.append(value_after_withdrawal)
        total_values.append(total_value)
        cumulative_dividends_list.append(cum_dividends)
        cumulative_withdrawal_list.append(cum_withdrawal)
        cumulative_interest_list.append(cum_interest)
        cash_etf_list.append(cash_etf)
        cash_reserve_list.append(cash_reserve)
        shares_history.append(shares.copy())

        if depleted_at is not None:
            break

    result = pd.DataFrame({
        'value_after_withdrawal': value_after_withdrawal_list,
        'total_portfolio_value': total_values,
        'cumulative_dividends': cumulative_dividends_list,
        'cumulative_withdrawals': cumulative_withdrawal_list,
        'cumulative_interest': cumulative_interest_list,
        'cash_etf': cash_etf_list,
        'cash_reserve': cash_reserve_list
    }, index=price_df.index[:len(value_after_withdrawal_list)])

    shares_df = pd.DataFrame(shares_history, index=price_df.index[:len(value_after_withdrawal_list)])
    result = pd.concat([result, shares_df], axis=1)

    val_df = shares_df.mul(price_df.loc[result.index]).add_prefix("val_")
    val_df['val_cash'] = result['cash_reserve']
    result  = result.join(val_df)

    return result, depleted_at

