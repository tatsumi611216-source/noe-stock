#!/usr/bin/env python3
"""
データ取得モジュール
yfinanceで日本株・米国株・暗号資産の価格履歴をSQLiteに保存する
ネットワーク制限がある環境では既存DBデータをそのまま活用する
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

import sqlite3
import time
import warnings
import logging
from datetime import datetime, timedelta
from pathlib import Path

# yfinanceの冗長なログを抑制
warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

import yfinance as yf
import pandas as pd

DB_PATH = Path(__file__).parent / "stock_data.db"

# 監視銘柄リスト
JP_STOCKS = [
    "7203.T",  # トヨタ
    "9984.T",  # ソフトバンクG
    "6758.T",  # ソニー
    "6861.T",  # キーエンス
    "8306.T",  # 三菱UFJ
    "9432.T",  # NTT
    "6501.T",  # 日立
    "8035.T",  # 東京エレクトロン
    "4063.T",  # 信越化学
    "7974.T",  # 任天堂
    "9433.T",  # KDDI
    "4543.T",  # テルモ
    "2914.T",  # JT
    "8411.T",  # みずほFG
    "6367.T",  # ダイキン
    "4519.T",  # 中外製薬
    "7267.T",  # ホンダ
    "6954.T",  # ファナック
    "3382.T",  # セブン&アイ
    "6902.T",  # デンソー
]

US_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "BRK-B", "JPM", "JNJ",
    "V", "PG", "MA", "HD", "CVX",
]

CRYPTO = [
    "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD",
]

ALL_SYMBOLS = JP_STOCKS + US_STOCKS + CRYPTO


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            symbol TEXT,
            date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            PRIMARY KEY (symbol, date)
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS fundamentals (
            symbol TEXT PRIMARY KEY,
            name TEXT,
            market TEXT,
            sector TEXT,
            pe_ratio REAL,
            pb_ratio REAL,
            roe REAL,
            market_cap REAL,
            updated_at TEXT
        )
    """)
    conn.commit()
    return conn


def check_db_freshness(conn):
    """DBの最新日付と銘柄数を返す"""
    row = conn.execute(
        "SELECT MAX(date), COUNT(DISTINCT symbol) FROM prices"
    ).fetchone()
    latest_date = row[0] or "なし"
    symbol_count = row[1] or 0
    return latest_date, symbol_count


def fetch_prices(conn, symbols, period="1y"):
    latest_before, _ = check_db_freshness(conn)
    print(f"\n📥 価格データ取得中 ({len(symbols)}銘柄)...")
    print(f"   既存DB最新日: {latest_before}")
    c = conn.cursor()
    ok, ng, skipped = 0, 0, 0

    for sym in symbols:
        try:
            tk = yf.Ticker(sym)
            # 差分取得：DBの最新日の翌日から今日まで
            existing = c.execute(
                "SELECT MAX(date) FROM prices WHERE symbol=?", (sym,)
            ).fetchone()[0]
            if existing:
                start = (datetime.strptime(existing, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
                today = datetime.now().strftime("%Y-%m-%d")
                if start > today:
                    skipped += 1
                    continue
                df = tk.history(start=start, end=today, auto_adjust=True)
            else:
                df = tk.history(period=period, auto_adjust=True)

            if df.empty:
                skipped += 1
                continue

            df.index = df.index.strftime("%Y-%m-%d")
            rows = [
                (sym, d, row["Open"], row["High"], row["Low"], row["Close"], row["Volume"])
                for d, row in df.iterrows()
            ]
            c.executemany(
                "INSERT OR REPLACE INTO prices VALUES (?,?,?,?,?,?,?)", rows
            )
            conn.commit()
            ok += 1
            print(f"  ✓ {sym:<12} +{len(rows)}日分")
            time.sleep(0.3)
        except Exception:
            ng += 1

    latest_after, sym_count = check_db_freshness(conn)
    if ok > 0:
        print(f"\n  新規取得: {ok}銘柄  スキップ(最新): {skipped}  失敗: {ng}")
    elif ng == len(symbols) - skipped:
        print(f"\n  ⚠️  ネットワーク制限により新規取得不可。既存DB({sym_count}銘柄 / {latest_after})を使用します。")
    else:
        print(f"\n  完了: {ok}成功 / {ng}失敗 / {skipped}スキップ")


def fetch_fundamentals(conn, symbols):
    print(f"\n📊 ファンダメンタルズ取得中...")
    c = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d")
    ok, ng = 0, 0

    for sym in symbols:
        if sym.endswith("-USD"):
            c.execute(
                "INSERT OR REPLACE INTO fundamentals VALUES (?,?,?,?,?,?,?,?,?)",
                (sym, sym.replace("-USD",""), "CRYPTO", "Crypto", None, None, None, None, now)
            )
            ok += 1
            continue
        try:
            info = yf.Ticker(sym).info
            if not info or info.get("regularMarketPrice") is None and not info.get("longName"):
                raise ValueError("empty info")
            market = "JP" if sym.endswith(".T") else "US"
            c.execute(
                "INSERT OR REPLACE INTO fundamentals VALUES (?,?,?,?,?,?,?,?,?)",
                (sym,
                 info.get("longName") or info.get("shortName", sym),
                 market,
                 info.get("sector", ""),
                 info.get("trailingPE"),
                 info.get("priceToBook"),
                 info.get("returnOnEquity"),
                 info.get("marketCap"),
                 now)
            )
            conn.commit()
            ok += 1
            time.sleep(0.5)
        except Exception:
            ng += 1

    if ng > 0 and ok <= len([s for s in symbols if s.endswith("-USD")]):
        print(f"  ⚠️  ファンダメンタルズ取得不可。既存データを使用します。")
    else:
        print(f"  完了: {ok}成功 / {ng}失敗")


def main():
    conn = init_db()
    latest, count = check_db_freshness(conn)
    print(f"📂 DB確認: {count}銘柄 / 最新日 {latest}")

    fetch_prices(conn, ALL_SYMBOLS)
    fetch_fundamentals(conn, ALL_SYMBOLS)
    conn.close()

    latest_final, count_final = check_db_freshness(init_db())
    print(f"\n✅ DB保存完了: {DB_PATH}")
    print(f"   {count_final}銘柄 / 最新日 {latest_final}")


if __name__ == "__main__":
    main()
