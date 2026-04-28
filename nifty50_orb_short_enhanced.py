# ============================================================
# NIFTY 50 - ORB SHORT STRATEGY (5-Year Backtest, Intraday)
# Dual timeframe rules:
#   - Primary decision window: first 30 minutes (09:15-09:45) ORB
#   - Execution interval: 5-minute candles for signals + filters
# Extra execution controls:
#   - Max 3 trades per day per symbol
#   - ADX trend filter, RSI mean-reversion filter, ATR exits, volume confirmation
# ============================================================

import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# Reuse these from the parent notebook/script scope:
# NIFTY_50_TICKERS, CAPITAL, YEARS_BACK, BROKERAGE,
# STT_RATE, TRANSACTION_RATE, GST_RATE, STAMP_DUTY_RATE

# Strategy controls
MAX_TRADES_PER_DAY = 3
ADX_TREND_THRESHOLD = 25
ADX_RANGE_THRESHOLD = 20
RSI_OVERBOUGHT = 70
ATR_TARGET_MULT = 0.75
ATR_STOP_MULT = 1.5
VOLUME_CONFIRM_MULT = 1.20
ORB_CANDLES = 6  # 6x5m = first 30 minutes


def calc_charges_short(entry, exit_, qty):
    """Charges model for short equity carry trade."""
    turnover = (entry + exit_) * qty
    brokerage = BROKERAGE * 2
    stt = entry * qty * STT_RATE
    txn = turnover * TRANSACTION_RATE
    gst = (brokerage + txn) * GST_RATE
    stamp = exit_ * qty * STAMP_DUTY_RATE
    return round(brokerage + stt + txn + gst + stamp, 2)


def add_intraday_indicators(df, rsi_period=14, adx_period=14, atr_period=14):
    """Compute RSI, ADX and ATR on 5-minute candles."""
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

    out["VolMA20"] = out["Volume"].rolling(20).mean()
    out["VolConfirm"] = out["Volume"] >= out["VolMA20"] * VOLUME_CONFIRM_MULT

    return out


def download_5m_history(ticker, years=5):
    """
    Download 5m history in chunks because Yahoo limits intraday lookback per request.
    """
    end = datetime.utcnow()
    start = end - timedelta(days=years * 365 + 5)

    chunks = []
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=59), end)
        try:
            part = yf.download(
                ticker,
                start=cur,
                end=nxt,
                interval="5m",
                auto_adjust=True,
                progress=False,
                prepost=False,
            )
            if not part.empty:
                if isinstance(part.columns, pd.MultiIndex):
                    part.columns = part.columns.get_level_values(0)
                part = part[["Open", "High", "Low", "Close", "Volume"]].dropna()
                chunks.append(part)
        except Exception:
            pass
        cur = nxt

    if not chunks:
        return None

    df = pd.concat(chunks).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    if len(df) < 200:
        return None
    return add_intraday_indicators(df)


def first_session_open_next_day(df, day):
    """Return next day's first 5m open after `day`, else None."""
    next_day = day + pd.Timedelta(days=1)
    while next_day <= df.index.max().normalize():
        day_slice = df[df.index.normalize() == next_day]
        if not day_slice.empty:
            return float(day_slice.iloc[0]["Open"]), day_slice.index[0]
        next_day += pd.Timedelta(days=1)
    return None, None


def simulate_intraday_orb_short(ticker, df):
    """Dual-timeframe ORB short simulation with max 3 trades/day."""
    trades = []
    by_day = list(df.groupby(df.index.normalize()))

    for day, day_df in by_day:
        if len(day_df) <= ORB_CANDLES + 2:
            continue

        orb_window = day_df.iloc[:ORB_CANDLES]
        orb_high = float(orb_window["High"].max())
        orb_low = float(orb_window["Low"].min())

        signal_df = day_df.iloc[ORB_CANDLES:].copy()
        trades_today = 0

        for i in range(1, len(signal_df)):
            if trades_today >= MAX_TRADES_PER_DAY:
                break

            cur = signal_df.iloc[i]
            prev = signal_df.iloc[i - 1]
            candle_time = signal_df.index[i]

            adx = float(cur["ADX"]) if not pd.isna(cur["ADX"]) else np.nan
            rsi = float(cur["RSI"]) if not pd.isna(cur["RSI"]) else np.nan
            atr = float(cur["ATR"]) if not pd.isna(cur["ATR"]) else np.nan
            vol_ok = bool(cur.get("VolConfirm", False))

            if np.isnan(adx) or np.isnan(rsi) or np.isnan(atr) or atr <= 0 or not vol_ok:
                continue

            trend_breakdown = (cur["Close"] < orb_low) and (adx > ADX_TREND_THRESHOLD)
            fade_breakout = (
                adx < ADX_RANGE_THRESHOLD
                and rsi > RSI_OVERBOUGHT
                and prev["Close"] > orb_high
                and cur["Close"] > orb_high
            )

            if not (trend_breakdown or fade_breakout):
                continue

            entry = float(cur["Close"])  # execute on 5m signal close
            target = entry - ATR_TARGET_MULT * atr
            stop = entry + ATR_STOP_MULT * atr
            qty = max(1, int(CAPITAL / entry))

            exit_price = None
            exit_time = None
            exit_mode = None

            post = day_df[day_df.index > candle_time]
            for _, bar in post.iterrows():
                if bar["Open"] <= target:
                    exit_price = float(bar["Open"])
                    exit_time = bar.name
                    exit_mode = "TargetGap"
                    break
                if bar["Open"] >= stop:
                    exit_price = float(bar["Open"])
                    exit_time = bar.name
                    exit_mode = "StopGap"
                    break
                if bar["Low"] <= target:
                    exit_price = float(target)
                    exit_time = bar.name
                    exit_mode = "Target"
                    break
                if bar["High"] >= stop:
                    exit_price = float(stop)
                    exit_time = bar.name
                    exit_mode = "Stop"
                    break

            # If no same-day ATR exit, cover next day's first 5m open.
            if exit_price is None:
                exit_price, exit_time = first_session_open_next_day(df, day)
                if exit_price is None:
                    continue
                exit_mode = "NextDayOpen"

            gross = round((entry - exit_price) * qty, 2)
            charges = calc_charges_short(entry, exit_price, qty)
            net = round(gross - charges, 2)

            trades.append(
                {
                    "Symbol": ticker.replace(".NS", ""),
                    "Date": day.strftime("%Y-%m-%d"),
                    "Entry Time": candle_time.strftime("%Y-%m-%d %H:%M"),
                    "Exit Time": pd.Timestamp(exit_time).strftime("%Y-%m-%d %H:%M"),
                    "ORB High": round(orb_high, 2),
                    "ORB Low": round(orb_low, 2),
                    "Entry Price": round(entry, 2),
                    "Exit Price": round(exit_price, 2),
                    "Qty": qty,
                    "ADX": round(adx, 2),
                    "RSI": round(rsi, 2),
                    "ATR": round(atr, 2),
                    "Volume Confirm": vol_ok,
                    "Target": round(target, 2),
                    "Stop Loss": round(stop, 2),
                    "Setup": "TrendBreakdown" if trend_breakdown else "FadeBreakout",
                    "Exit Mode": exit_mode,
                    "gross": gross,
                    "charges": charges,
                    "net": net,
                    "Type": "Short",
                }
            )
            trades_today += 1

    return trades


def run_short_backtest():
    sep = "=" * 90
    print(sep)
    print("NIFTY 50 - ORB SHORT STRATEGY (DUAL TIMEFRAME)")
    print("5-MIN EXECUTION + 30-MIN ORB + MAX 3 TRADES/DAY")
    print(sep)
    print(f"Capital per stock : Rs.{CAPITAL:,}")
    print(f"Total stocks      : {len(NIFTY_50_TICKERS)}")
    print(f"Period            : {YEARS_BACK} years")
    print(f"Rules             : ADX>{ADX_TREND_THRESHOLD}, RSI>{RSI_OVERBOUGHT}, VolConfirm x{VOLUME_CONFIRM_MULT}")
    print(f"Risk              : Target {ATR_TARGET_MULT}xATR / Stop {ATR_STOP_MULT}xATR")
    print(sep)

    all_trades = []
    summary = []

    for i, ticker in enumerate(NIFTY_50_TICKERS, 1):
        print(f"[{i:2d}/{len(NIFTY_50_TICKERS)}] {ticker:<20s}", end="")
        df = download_5m_history(ticker, years=YEARS_BACK)
        if df is None:
            print("  No 5m data")
            continue

        trades = simulate_intraday_orb_short(ticker, df)
        print(f"  {len(df):,} candles  {len(trades)} trades")

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
                    "Win Rate %": round((wins / len(trades)) * 100, 2),
                    "Return %": round((pnl / CAPITAL) * 100, 2),
                }
            )

    trades_df = pd.DataFrame(all_trades)
    summary_df = pd.DataFrame(summary)

    fname = "nifty50_short_tradelog_5years_dualtimeframe.xlsx"
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

        print(f"Total Trades    : {total_trades:,}")
        print(f"Gross P&L       : Rs.{total_gross:>15,.2f}")
        print(f"Total Charges   : Rs.{total_charges:>15,.2f}")
        print(f"Net P&L         : Rs.{total_net:>15,.2f}")
        print(f"Win Rate        : {(winning / total_trades) * 100:.2f}%")
        print(f"Avg Trade (Net) : Rs.{total_net / total_trades:>12,.2f}")
        print(f"Annual Average  : Rs.{total_net / YEARS_BACK:>12,.2f}")

        if not summary_df.empty:
            print("\nTop 10 Stocks by Net P&L (Short):")
            print(summary_df.nlargest(10, "Total P&L").to_string(index=False))

            print("\nBottom 5 Stocks by Net P&L (Short):")
            print(summary_df.nsmallest(5, "Total P&L").to_string(index=False))
    else:
        print("No trades generated.")

    print("\n" + sep + "\nBACKTEST COMPLETE! (SHORT)\n" + sep)
    return trades_df, summary_df


# Uncomment to run standalone:
# short_trades_df, short_summary_df = run_short_backtest()
