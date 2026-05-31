#!/usr/bin/env python3
"""
当日の銘柄推薦スクリプト
EV上位銘柄をランキング表示 + HTMLレポート出力
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

import sqlite3
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path

DB_PATH    = Path(__file__).parent / "stock_data.db"
MODEL_PATH = Path(__file__).parent / "model_xgb.joblib"
ISO_PATH   = Path(__file__).parent / "model_iso.joblib"
OUT_DIR    = Path(__file__).parent.parent / "outputs"

from calc_features import FEATURE_COLS, build_all_features
from calc_ev import calc_stock_ev, kelly_bet_size

OUT_DIR.mkdir(exist_ok=True)
TODAY = datetime.now().strftime("%Y-%m-%d")


def load_model():
    if not MODEL_PATH.exists():
        print("⚠️  モデルが見つかりません。python train_model.py を先に実行してください。")
        print("   （初回のみ。データが少ない場合はルールベースで代替します）")
        return None, None
    model = joblib.load(MODEL_PATH)
    iso   = joblib.load(ISO_PATH)
    return model, iso


def rule_based_score(row) -> float:
    """
    モデルがない場合のルールベーススコア
    複数の指標を合成してP_upを擬似的に算出
    """
    score = 0.5  # ベースライン

    # RSIが40〜60のゾーン（過熱・売られすぎでない）
    rsi = row.get("rsi_14", 50)
    if 35 <= rsi <= 60:
        score += 0.05
    elif rsi < 30:
        score += 0.08  # 売られすぎからの反発期待
    elif rsi > 75:
        score -= 0.08

    # MACDヒストグラムがプラスに転換（買いシグナル）
    if row.get("macd_hist", 0) > 0 and row.get("macd", 0) > row.get("macd_signal", 0):
        score += 0.06

    # 移動平均が上向き
    if row.get("sma20_slope", 0) > 0:
        score += 0.03
    if row.get("sma60_slope", 0) > 0:
        score += 0.03

    # 直近20日モメンタムがプラスだが過熱していない
    mom = row.get("momentum_20d", 0)
    if 0.02 <= mom <= 0.15:
        score += 0.05
    elif mom > 0.20:
        score -= 0.03  # 上昇しすぎは利確リスク

    # 出来高急増（ブレイクアウト可能性）
    if row.get("volume_ratio", 1) > 1.5:
        score += 0.04

    # 52週高値付近（強いトレンド）
    if row.get("high_52w_ratio", 0.9) > 0.95:
        score += 0.03

    return max(0.1, min(0.9, score))


def predict(conn, model, iso):
    print(f"\n🔮 予測実行: {TODAY}")
    df = build_all_features(conn)
    if df.empty:
        print("⚠️  特徴量データなし。fetch_data.py を実行してください。")
        return []

    results = []
    for _, row in df.iterrows():
        if model is not None:
            X = pd.DataFrame([row])[FEATURE_COLS].fillna(0).values
            p_raw = model.predict_proba(X)[:, 1][0]
            p_up  = float(iso.predict([p_raw])[0])
        else:
            p_up = rule_based_score(row.to_dict())

        ev_result = calc_stock_ev(p_up)
        kelly = kelly_bet_size(p_up, ev_result["target_return"], ev_result["stop_loss"])

        # ファンダメンタルズ取得
        fund = conn.execute(
            "SELECT name, market, sector FROM fundamentals WHERE symbol=?",
            (row["symbol"],)
        ).fetchone()
        name   = fund[0] if fund else row["symbol"]
        market = fund[1] if fund else "?"
        sector = fund[2] if fund else ""

        results.append({
            "symbol":     row["symbol"],
            "name":       name,
            "market":     market,
            "sector":     sector,
            "close":      row.get("close", 0),
            "p_up":       p_up,
            "ev":         ev_result["ev"],
            "label":      ev_result["label"],
            "kelly_pct":  round(kelly * 100, 2),
            "rsi_14":     row.get("rsi_14", 0),
            "macd_hist":  row.get("macd_hist", 0),
            "momentum_20d": row.get("momentum_20d", 0),
        })

    return sorted(results, key=lambda x: x["ev"], reverse=True)


def print_report(results):
    top = [r for r in results if r["ev"] >= 0.0][:20]
    print("\n" + "=" * 80)
    print(f"🏆 Noe Stock AI — 推薦銘柄レポート {TODAY}")
    print("=" * 80)

    for market in ["JP", "US", "CRYPTO"]:
        market_top = [r for r in top if r["market"] == market][:5]
        if not market_top:
            continue
        labels = {"JP": "🇯🇵 日本株", "US": "🇺🇸 米国株", "CRYPTO": "₿ 暗号資産"}
        print(f"\n{labels[market]}")
        print(f"  {'銘柄':<12} {'名称':<20} {'価格':>10} {'P_up':>6} {'EV':>7} {'kelly%':>7}  {'評価'}")
        print("  " + "-" * 75)
        for r in market_top:
            close_str = f"{r['close']:>10,.1f}"
            print(f"  {r['symbol']:<12} {r['name'][:18]:<20} {close_str} "
                  f"{r['p_up']:>5.1%} {r['ev']:>+6.3f} {r['kelly_pct']:>6.1f}%  {r['label']}")

    print("\n" + "=" * 80)
    print("【見方】EV = 期待値  P_up = 上昇確率  kelly% = 推奨ポジションサイズ（総資金比）")
    print("       EV+0.05以上が購入候補。kelly%は1回あたりの推奨投資比率。")
    print("=" * 80)


def generate_html(results):
    top = [r for r in results if r["ev"] >= 0.0][:30]
    rows_html = ""
    for r in top:
        ev_color = "#22c55e" if r["ev"] >= 0.05 else ("#86efac" if r["ev"] >= 0 else "#fca5a5")
        star = r["label"].split()[0] if r["label"].startswith("★") else ""
        rows_html += f"""
        <tr>
          <td>{r['symbol']}</td>
          <td>{r['name'][:20]}</td>
          <td>{r['market']}</td>
          <td>{r['sector'][:15]}</td>
          <td style="text-align:right">{r['close']:,.1f}</td>
          <td style="text-align:right">{r['p_up']:.1%}</td>
          <td style="text-align:right; color:{ev_color}; font-weight:bold">{r['ev']:+.3f}</td>
          <td style="text-align:right">{r['kelly_pct']:.1f}%</td>
          <td>{r['rsi_14']:.1f}</td>
          <td style="color:{'#22c55e' if r['macd_hist']>0 else '#ef4444'}">{r['macd_hist']:+.4f}</td>
          <td style="color:{'#22c55e' if r['momentum_20d']>0 else '#ef4444'}">{r['momentum_20d']:+.1%}</td>
          <td>{star}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <title>Noe Stock AI — {TODAY}</title>
  <style>
    body {{ font-family: 'Segoe UI', sans-serif; background:#0f172a; color:#e2e8f0; padding:20px; }}
    h1 {{ color:#38bdf8; }}
    table {{ border-collapse:collapse; width:100%; font-size:13px; }}
    th {{ background:#1e293b; color:#94a3b8; padding:8px; text-align:left; border-bottom:2px solid #334155; }}
    td {{ padding:7px 8px; border-bottom:1px solid #1e293b; }}
    tr:hover {{ background:#1e293b; }}
    .badge {{ display:inline-block; padding:2px 8px; border-radius:999px; font-size:11px; }}
    .jp {{ background:#1d4ed8; }} .us {{ background:#7c3aed; }} .crypto {{ background:#d97706; }}
  </style>
</head>
<body>
  <h1>🏆 Noe Stock AI — {TODAY}</h1>
  <p style="color:#64748b">EV（期待値）上位銘柄。EV+0.05以上が購入候補。</p>
  <table>
    <thead>
      <tr>
        <th>銘柄</th><th>名称</th><th>市場</th><th>セクター</th>
        <th>価格</th><th>P_up</th><th>EV</th><th>Kelly%</th>
        <th>RSI</th><th>MACD</th><th>20日騰落</th><th>評価</th>
      </tr>
    </thead>
    <tbody>{rows_html}</tbody>
  </table>
  <p style="color:#475569; margin-top:20px; font-size:12px">
    Generated by Noe Stock AI | {TODAY} | 投資判断は自己責任でお願いします
  </p>
</body>
</html>"""

    out_path = OUT_DIR / f"stock_report_{TODAY}.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"\n📄 HTMLレポート: {out_path}")
    return out_path


def main():
    print("=" * 80)
    print(f"🚀 Noe Stock AI — Daily Run {TODAY}")
    print("=" * 80)

    conn = sqlite3.connect(DB_PATH)
    model, iso = load_model()
    results = predict(conn, model, iso)
    conn.close()

    if not results:
        return

    print_report(results)
    generate_html(results)


if __name__ == "__main__":
    main()
