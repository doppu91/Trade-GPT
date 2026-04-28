# ============================================================
# NIFTY 50 - ORB SHORT STRATEGY  (5-Year Backtest)
# Base logic retained + requested enhancements:
# 1) ADX trend-strength filter (breakdown trades only when ADX > 25)
# 2) Mean-reversion fade setup at ORB top with RSI > 70
# 3) ATR-based dynamic target/stop (target 0.75x ATR, stop 1.5x ATR)
# 4) Two-candle 5m confirmation above first 30m high (for fade entries)
# ============================================================

import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── reuse constants already defined ─────────────────────────────────────────
# (NIFTY_50_TICKERS, CAPITAL, YEARS_BACK, BROKERAGE, rates all in scope)
# If running standalone, uncomment and fill:
# NIFTY_50_TICKERS = [ ... ]
# CAPITAL = 300000; YEARS_BACK = 5; BROKERAGE = 20
# STT_RATE = 0.00025; TRANSACTION_RATE = 0.00345
# GST_RATE = 0.18; STAMP_DUTY_RATE = 0.00015

ADX_TREND_THRESHOLD = 25
ADX_RANGE_THRESHOLD = 20
RSI_OVERBOUGHT = 70
ATR_TARGET_MULT = 0.75
ATR_STOP_MULT = 1.5


def calc_charges_short(entry, exit_, qty):
    """
    For SHORT trades:
      - STT on SELL (entry) leg for equity short carry model.
      - Stamp duty on BUY (cover) leg.
    """

    def _as_fraction(rate):
        # Supports either decimal fraction (0.00025) or percent-like input (0.025).
        return rate / 100 if rate > 0.001 else rate

    stt_rate = _as_fraction(STT_RATE)
    txn_rate = _as_fraction(TRANSACTION_RATE)
    stamp_rate = _as_fraction(STAMP_DUTY_RATE)

    turnover = (entry + exit_) * qty
    brokerage = BROKERAGE * 2
    stt = entry * qty * stt_rate
    txn = turnover * txn_rate
    gst = (brokerage + txn) * GST_RATE
    stamp = exit_ * qty * stamp_rate
    return round(brokerage + stt + txn + gst + stamp, 2)


def add_daily_indicators(df, adx_period=14, rsi_period=14, atr_period=14):
    """Adds ADX, RSI, ATR columns on daily OHLC data."""
    out = df.copy()

    prev_close = out["Close"].shift(1)
    tr = pd.concat(
        [
            out["High"] - out["Low"],
            (out["High"] - prev_close).abs(),
            (out["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["ATR"] = tr.rolling(atr_period).mean()

    delta = out["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / rsi_period, adjust=False, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1 / rsi_period, adjust=False, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["RSI"] = 100 - (100 / (1 + rs))

    up_move = out["High"].diff()
    down_move = -out["Low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr_smooth = tr.ewm(alpha=1 / adx_period, adjust=False, min_periods=adx_period).mean()
    plus_di = 100 * pd.Series(plus_dm, index=out.index).ewm(
        alpha=1 / adx_period, adjust=False, min_periods=adx_period
    ).mean() / tr_smooth
    minus_di = 100 * pd.Series(minus_dm, index=out.index).ewm(
        alpha=1 / adx_period, adjust=False, min_periods=adx_period
    ).mean() / tr_smooth

    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    out["ADX"] = dx.ewm(alpha=1 / adx_period, adjust=False, min_periods=adx_period).mean()

    return out


def get_data_short(ticker, years=5):
    end = datetime.today()
    start = end - timedelta(days=years * 365 + 45)
    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if df.empty or len(df) < 80:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        return add_daily_indicators(df)
    except Exception:
        return None


def two_candle_confirmation_above_30m_high(ticker, day_ts):
    """
    Returns True if there are 2 consecutive 5m closes above first-30-minute high.
    """
    day_start = pd.Timestamp(day_ts).normalize()
    day_end = day_start + pd.Timedelta(days=1)

    try:
        intra = yf.download(
            ticker,
            start=day_start,
            end=day_end,
            interval="5m",
            auto_adjust=True,
            progress=False,
            prepost=False,
        )
        if intra.empty:
            return False
        if isinstance(intra.columns, pd.MultiIndex):
            intra.columns = intra.columns.get_level_values(0)
        intra = intra[["Open", "High", "Low", "Close"]].dropna()
        if len(intra) < 8:
            return False

        first_30m_high = intra["High"].iloc[:6].max()
        closes = intra["Close"].to_numpy()

        for i in range(1, len(closes)):
            if closes[i - 1] > first_30m_high and closes[i] > first_30m_high:
                return True
        return False
    except Exception:
        return False


def apply_atr_exit_short(entry_price, next_open, next_high, next_low, atr):
    """
    Short exit model:
      target = entry - 0.75*ATR
      stop   = entry + 1.5*ATR
    Uses next-day OHLC as approximation for whether target/stop was hit.
    """
    target = entry_price - ATR_TARGET_MULT * atr
    stop = entry_price + ATR_STOP_MULT * atr

    # Gap at open first
    if next_open <= target:
        return next_open, "TargetGap", target, stop
    if next_open >= stop:
        return next_open, "StopGap", target, stop

    # Day-range hit approximation
    if next_low <= target:
        return target, "Target", target, stop
    if next_high >= stop:
        return stop, "Stop", target, stop

    return next_open, "Open", target, stop


def simulate_orb_short(ticker, df):
    """
    Base-style ORB short with enhancements:
      - Trend breakdown short: Close < ORB Low and ADX > 25
      - Mean reversion short: price at ORB High, RSI > 70, ADX < 20,
                              and two-candle 5m confirmation above 30m high
      - Exit using ATR dynamic levels (next-day OHLC proxy)
    """
    trades = []

    for i in range(20, len(df) - 1):
        row = df.iloc[i]
        o = float(row["Open"])
        h = float(row["High"])
        l = float(row["Low"])
        c = float(row["Close"])
        adx = float(row["ADX"]) if not pd.isna(row["ADX"]) else np.nan
        rsi = float(row["RSI"]) if not pd.isna(row["RSI"]) else np.nan
        atr = float(row["ATR"]) if not pd.isna(row["ATR"]) else np.nan

        if o <= 0 or np.isnan(adx) or np.isnan(rsi) or np.isnan(atr):
            continue

        orb_high = o + (h - o) * 0.15
        orb_low = o - (o - l) * 0.15

        trend_breakdown = c < orb_low and adx > ADX_TREND_THRESHOLD
        fade_breakout = (
            c >= orb_high
            and rsi > RSI_OVERBOUGHT
            and adx < ADX_RANGE_THRESHOLD
            and two_candle_confirmation_above_30m_high(ticker, df.index[i])
        )

        if not (trend_breakdown or fade_breakout):
            continue

        entry_price = orb_low if trend_breakdown else orb_high
        next_day = df.iloc[i + 1]
        exit_price, exit_mode, target, stop = apply_atr_exit_short(
            entry_price=entry_price,
            next_open=float(next_day["Open"]),
            next_high=float(next_day["High"]),
            next_low=float(next_day["Low"]),
            atr=atr,
        )

        qty = max(1, int(CAPITAL / entry_price))
        gross = round((entry_price - exit_price) * qty, 2)
        charges = calc_charges_short(entry_price, exit_price, qty)
        net = round(gross - charges, 2)

        trades.append(
            {
                "Symbol": ticker.replace(".NS", ""),
                "Date": df.index[i].strftime("%Y-%m-%d"),
                "Entry Price": round(entry_price, 2),
                "Exit Price": round(exit_price, 2),
                "Qty": qty,
                "Stop Loss": round(stop, 2),
                "Target": round(target, 2),
                "ADX": round(adx, 2),
                "RSI": round(rsi, 2),
                "ATR": round(atr, 2),
                "Setup": "TrendBreakdown" if trend_breakdown else "FadeBreakout",
                "Exit Mode": exit_mode,
                "gross": gross,
                "charges": charges,
                "net": net,
                "Type": "Short",
            }
        )

    return trades


def run_short_backtest():
    sep = "=" * 80
    print(sep)
    print("NIFTY 50 - ORB SHORT STRATEGY")
    print("5 YEAR BACKTESTING WITH TRADE LOG")
    print(sep)
    print(f"Capital per stock : Rs.{CAPITAL:,}")
    print(f"Total stocks      : {len(NIFTY_50_TICKERS)}")
    print(f"Period            : {YEARS_BACK} years")
    print(
        f"Filters           : ADX>{ADX_TREND_THRESHOLD} trend, "
        f"ADX<{ADX_RANGE_THRESHOLD} + RSI>{RSI_OVERBOUGHT} fade"
    )
    print(
        f"ATR exits         : Target {ATR_TARGET_MULT}xATR, "
        f"Stop {ATR_STOP_MULT}xATR"
    )
    print(sep)

    all_trades = []
    summary = []

    for i, ticker in enumerate(NIFTY_50_TICKERS, 1):
        print(f"[{i:2d}/{len(NIFTY_50_TICKERS)}] {ticker:<20s}", end="")
        df = get_data_short(ticker, years=YEARS_BACK)
        if df is None:
            print("  No data")
            continue

        trades = simulate_orb_short(ticker, df)
        print(f"  {len(df)} days  {len(trades)} trades")

        if trades:
            all_trades.extend(trades)
            pnl = sum(t["net"] for t in trades)
            gross = sum(t["gross"] for t in trades)
            charges = sum(t["charges"] for t in trades)
            wins = sum(1 for t in trades if t["net"] > 0)
            summary.append(
                {
                    "Symbol": ticker.replace(".NS", ""),
                    "Trades": len(trades),
                    "Gross P&L": round(gross, 2),
                    "Charges": round(charges, 2),
                    "Total P&L": round(pnl, 2),
                    "Win Trades": wins,
                    "Win Rate %": round(wins / len(trades) * 100, 2),
                    "Return %": round(pnl / CAPITAL * 100, 2),
                }
            )

    trades_df = pd.DataFrame(all_trades)
    summary_df = pd.DataFrame(summary)

    fname = "nifty50_short_tradelog_5years.xlsx"
    print("\n" + sep + "\nSAVING RESULTS...\n" + sep)
    try:
        with pd.ExcelWriter(fname, engine="openpyxl") as writer:
            if not trades_df.empty:
                trades_df.to_excel(writer, sheet_name="All Trades", index=False)
            if not summary_df.empty:
                summary_df.to_excel(writer, sheet_name="Stock Summary", index=False)
                trades_df["Month"] = pd.to_datetime(trades_df["Date"]).dt.to_period("M").astype(str)
                monthly = trades_df.groupby("Month").agg(
                    Trades=("net", "count"), Net_PnL=("net", "sum")
                ).reset_index()
                monthly.to_excel(writer, sheet_name="Monthly Breakdown", index=False)
        print(f"Saved: {fname}")
    except Exception as e:
        print(f"Excel save failed: {e}")

    print("\n" + sep + "\nPORTFOLIO SUMMARY (SHORT)\n" + sep)

    if not trades_df.empty:
        total_trades = len(trades_df)
        total_gross = trades_df["gross"].sum()
        total_charges = trades_df["charges"].sum()
        total_net = trades_df["net"].sum()
        winning = (trades_df["net"] > 0).sum()
        win_rate = winning / total_trades * 100

        print(f"Total Trades    : {total_trades:,}")
        print(f"Gross P&L       : Rs.{total_gross:>15,.2f}")
        print(f"Total Charges   : Rs.{total_charges:>15,.2f}")
        print(f"Net P&L         : Rs.{total_net:>15,.2f}")
        print(f"Win Rate        : {win_rate:.2f}%")
        print(f"Avg Trade (Net) : Rs.{total_net / total_trades:>12,.2f}")

        trade_span_days = (
            pd.to_datetime(trades_df["Date"]).max() - pd.to_datetime(trades_df["Date"]).min()
        ).days + 1
        span_years = max(trade_span_days / 365.25, 1 / 12)
        span_months = max(trade_span_days / 30.44, 1.0)
        trading_days = max(trade_span_days * (252 / 365.25), 1.0)

        print(f"Annual Average  : Rs.{total_net / span_years:>12,.2f}")
        print(f"Monthly Average : Rs.{total_net / span_months:>12,.2f}")
        print(f"Daily Average   : Rs.{total_net / trading_days:>12,.2f}")

        if not summary_df.empty:
            print("\nTop 10 Stocks by Net P&L (Short):")
            top10 = summary_df.nlargest(10, "Total P&L")[
                ["Symbol", "Trades", "Total P&L", "Win Rate %", "Return %"]
            ]
            print(top10.to_string(index=False))

            print("\nBottom 5 Stocks by Net P&L (Short):")
            bot5 = summary_df.nsmallest(5, "Total P&L")[
                ["Symbol", "Trades", "Total P&L", "Win Rate %", "Return %"]
            ]
            print(bot5.to_string(index=False))

        disp_cols = [
            "Symbol",
            "Date",
            "Setup",
            "Entry Price",
            "Exit Price",
            "Qty",
            "ADX",
            "RSI",
            "ATR",
            "gross",
            "charges",
            "net",
        ]
        print("\nSample Trades (First 15):")
        print(trades_df[disp_cols].head(15).to_string(index=False))
    else:
        print("No trades generated.")

    print("\n" + sep + "\nBACKTEST COMPLETE! (SHORT)\n" + sep)
    return trades_df, summary_df


# ── RUN ───────────────────────────────────────────────────────────────────────
# short_trades_df, short_summary_df = run_short_backtest()
