#!/usr/bin/env python3
"""
特徴量計算モジュール
テクニカル指標 + ファンダメンタルズ → XGBoost用特徴量DataFrame
"""
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path

DB_PATH = Path(__file__).parent / "stock_data.db"

FEATURE_COLS = [
    "rsi_14", "rsi_5",
    "macd", "macd_signal", "macd_hist",
    "bb_width", "bb_position",
    "sma20_slope", "sma60_slope",
    "volume_ratio",
    "momentum_5d", "momentum_20d",
    "high_52w_ratio", "low_52w_ratio",
    "volatility_20d",
    "pe_ratio_norm", "pb_ratio_norm", "roe_norm",
    "market_idx",
]


def _rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _macd(series, fast=12, slow=26, signal=9):
    ema_f = series.ewm(span=fast, adjust=False).mean()
    ema_s = series.ewm(span=slow, adjust=False).mean()
    macd = ema_f - ema_s
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig, macd - sig


def calc_features_for_symbol(conn, symbol, as_of_date=None):
    """1銘柄の特徴量を計算してdictで返す"""
    c = conn.cursor()
    rows = c.execute(
        "SELECT date, open, high, low, close, volume FROM prices WHERE symbol=? ORDER BY date",
        (symbol,)
    ).fetchall()

    if len(rows) < 60:
        return None

    df = pd.DataFrame(rows, columns=["date","open","high","low","close","volume"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    if as_of_date:
        df = df[df.index <= pd.Timestamp(as_of_date)]

    if len(df) < 60:
        return None

    close = df["close"]
    volume = df["volume"]

    # RSI
    rsi14 = _rsi(close, 14).iloc[-1]
    rsi5  = _rsi(close, 5).iloc[-1]

    # MACD
    macd, sig, hist = _macd(close)
    macd_v  = macd.iloc[-1]
    sig_v   = sig.iloc[-1]
    hist_v  = hist.iloc[-1]

    # ボリンジャーバンド
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_width    = ((bb_upper - bb_lower) / sma20).iloc[-1]
    bb_position = ((close - bb_lower) / (bb_upper - bb_lower)).iloc[-1]

    # 移動平均の傾き（直近5日で正規化）
    sma60 = close.rolling(60).mean()
    sma20_slope = (sma20.iloc[-1] - sma20.iloc[-6]) / sma20.iloc[-6] if len(sma20.dropna()) >= 6 else 0.0
    sma60_slope = (sma60.iloc[-1] - sma60.iloc[-6]) / sma60.iloc[-6] if len(sma60.dropna()) >= 6 else 0.0

    # 出来高比率（直近1日 vs 20日平均）
    vol_ma20 = volume.rolling(20).mean().iloc[-1]
    volume_ratio = volume.iloc[-1] / vol_ma20 if vol_ma20 > 0 else 1.0

    # モメンタム
    momentum_5d  = (close.iloc[-1] - close.iloc[-6])  / close.iloc[-6]  if len(close) >= 6  else 0.0
    momentum_20d = (close.iloc[-1] - close.iloc[-21]) / close.iloc[-21] if len(close) >= 21 else 0.0

    # 52週高値・安値比率
    c252 = close.iloc[-252:] if len(close) >= 252 else close
    high_52w_ratio = close.iloc[-1] / c252.max()
    low_52w_ratio  = close.iloc[-1] / c252.min()

    # ボラティリティ
    volatility_20d = close.pct_change().rolling(20).std().iloc[-1]

    # ファンダメンタルズ
    row = conn.execute(
        "SELECT pe_ratio, pb_ratio, roe, market FROM fundamentals WHERE symbol=?",
        (symbol,)
    ).fetchone()
    pe = (row[0] or 0.0) if row else 0.0
    pb = (row[1] or 0.0) if row else 0.0
    roe = (row[2] or 0.0) if row else 0.0
    market = (row[3] or "US") if row else "US"
    market_idx = {"JP": 0, "US": 1, "CRYPTO": 2}.get(market, 1)

    # 正規化（対数変換で外れ値を抑制）
    pe_norm  = np.log1p(max(0, pe))
    pb_norm  = np.log1p(max(0, pb))
    roe_norm = float(roe) if roe else 0.0

    return {
        "symbol": symbol,
        "date": df.index[-1].strftime("%Y-%m-%d"),
        "close": float(close.iloc[-1]),
        "rsi_14": float(rsi14),
        "rsi_5":  float(rsi5),
        "macd": float(macd_v),
        "macd_signal": float(sig_v),
        "macd_hist": float(hist_v),
        "bb_width": float(bb_width),
        "bb_position": float(bb_position),
        "sma20_slope": float(sma20_slope),
        "sma60_slope": float(sma60_slope),
        "volume_ratio": float(volume_ratio),
        "momentum_5d":  float(momentum_5d),
        "momentum_20d": float(momentum_20d),
        "high_52w_ratio": float(high_52w_ratio),
        "low_52w_ratio":  float(low_52w_ratio),
        "volatility_20d": float(volatility_20d),
        "pe_ratio_norm": float(pe_norm),
        "pb_ratio_norm": float(pb_norm),
        "roe_norm": float(roe_norm),
        "market_idx": market_idx,
    }


def build_all_features(conn, as_of_date=None):
    symbols = [r[0] for r in conn.execute("SELECT DISTINCT symbol FROM prices").fetchall()]
    records = []
    for sym in symbols:
        feat = calc_features_for_symbol(conn, sym, as_of_date)
        if feat:
            records.append(feat)
    return pd.DataFrame(records)


if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    df = build_all_features(conn)
    print(df[["symbol","date","rsi_14","macd_hist","momentum_20d"]].to_string())
    conn.close()
