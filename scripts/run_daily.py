#!/usr/bin/env python3
"""
1コマンド全自動 — Noe Stock Daily Run
=====================================
python run_daily.py              # 今日のデータ更新 + 予測
python run_daily.py --no-fetch   # データ更新スキップ（予測のみ）
python run_daily.py --train      # モデル再訓練も行う
"""
import sys
import argparse
import sqlite3
from datetime import datetime
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SCRIPTS_DIR = Path(__file__).parent
DB_PATH = SCRIPTS_DIR / "stock_data.db"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true", help="データ取得スキップ")
    ap.add_argument("--train",    action="store_true", help="モデル再訓練する")
    args = ap.parse_args()

    today = datetime.now().strftime("%Y-%m-%d")
    now_h = datetime.now().hour

    print("=" * 70)
    print(f"🚀 Noe Stock AI — Daily Run {today}")
    print("=" * 70)

    # 日本市場の状態確認
    if 9 <= now_h < 15:
        print(f"⚡ 東証取引時間中（{now_h}時）— リアルタイムに近いデータを使用")
    elif now_h < 9:
        print(f"🌙 東証開場前（{now_h}時）— 前日終値で予測します")
    else:
        print(f"🌇 東証終了後（{now_h}時）— 本日終値で予測します")

    # 1. データ取得
    if not args.no_fetch:
        print("\n" + "─" * 70)
        print("STEP 1: データ取得")
        print("─" * 70)
        sys.path.insert(0, str(SCRIPTS_DIR))
        from fetch_data import init_db, fetch_prices, fetch_fundamentals, ALL_SYMBOLS
        conn = init_db()

        # 差分取得（最新1週間のみ）
        fetch_prices(conn, ALL_SYMBOLS, period="5d")

        # ファンダメンタルズは週1回だけ更新
        fund_updated = conn.execute(
            "SELECT MIN(updated_at) FROM fundamentals"
        ).fetchone()[0]
        if not fund_updated or fund_updated < (datetime.now().strftime("%Y-%m-%d")):
            fetch_fundamentals(conn, ALL_SYMBOLS)

        conn.close()
    else:
        print("\n⏭  データ取得スキップ")

    # 2. モデル訓練（オプション）
    if args.train or not (SCRIPTS_DIR / "model_xgb.joblib").exists():
        print("\n" + "─" * 70)
        print("STEP 2: モデル訓練")
        print("─" * 70)
        from train_model import train
        conn = sqlite3.connect(DB_PATH)
        train(conn)
        conn.close()
    else:
        print("\n⏭  モデル訓練スキップ（既存モデルを使用）")

    # 3. 予測 + レポート生成
    print("\n" + "─" * 70)
    print("STEP 3: 予測 + レポート生成")
    print("─" * 70)
    from predict_today import main as predict_main
    predict_main()

    print("\n" + "=" * 70)
    print("✅ 完了")
    print("=" * 70)


if __name__ == "__main__":
    main()
