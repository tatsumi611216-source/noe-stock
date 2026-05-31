#!/usr/bin/env python3
"""
XGBoostモデル訓練
「20日後に+5%以上上昇したか」を予測する分類モデル
ウォークフォワード検証で過学習を防ぐ
"""
import sqlite3
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from pathlib import Path
import joblib

DB_PATH  = Path(__file__).parent / "stock_data.db"
MODEL_PATH = Path(__file__).parent / "model_xgb.joblib"
ISO_PATH   = Path(__file__).parent / "model_iso.joblib"

from calc_features import FEATURE_COLS, calc_features_for_symbol

FORWARD_DAYS = 20   # 何日後を予測するか
TARGET_RETURN = 0.05  # 上昇定義（+5%以上）


def build_training_data(conn):
    """全銘柄の過去データから訓練用DataFrame生成"""
    symbols = [r[0] for r in conn.execute("SELECT DISTINCT symbol FROM prices").fetchall()]
    records = []

    for sym in symbols:
        rows = conn.execute(
            "SELECT date, close FROM prices WHERE symbol=? ORDER BY date", (sym,)
        ).fetchall()
        if len(rows) < 100:
            continue

        dates = [r[0] for r in rows]
        closes = [r[1] for r in rows]

        for i in range(60, len(dates) - FORWARD_DAYS - 1):
            as_of = dates[i]
            future_close = closes[i + FORWARD_DAYS]
            current_close = closes[i]
            if current_close <= 0:
                continue

            feat = calc_features_for_symbol(conn, sym, as_of_date=as_of)
            if feat is None:
                continue

            y = 1 if (future_close - current_close) / current_close >= TARGET_RETURN else 0
            feat["y_up"] = y
            records.append(feat)

    return pd.DataFrame(records)


def train(conn):
    print("📊 訓練データ生成中（時間がかかります）...")
    df = build_training_data(conn)
    if len(df) < 200:
        print(f"⚠️  データ不足: {len(df)}件（最低200件必要）")
        print("   先に python fetch_data.py を実行してください")
        return

    print(f"  サンプル数: {len(df)}件  正例率: {df['y_up'].mean():.1%}")

    df = df.sort_values("date").reset_index(drop=True)
    split = int(len(df) * 0.8)
    train_df = df.iloc[:split]
    val_df   = df.iloc[split:]

    X_train = train_df[FEATURE_COLS].fillna(0).values
    y_train = train_df["y_up"].values
    X_val   = val_df[FEATURE_COLS].fillna(0).values
    y_val   = val_df["y_up"].values

    print("\n🤖 XGBoost訓練中...")
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        early_stopping_rounds=20,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)], verbose=50)

    p_val = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, p_val)
    print(f"\n  Validation AUC: {auc:.4f}")

    # アイソトニック回帰でキャリブレーション（競馬システムと同じ）
    p_train_all = model.predict_proba(X_train)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_train_all, y_train)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(iso, ISO_PATH)
    print(f"\n✅ モデル保存: {MODEL_PATH}")

    # 特徴量重要度
    importance = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    print("\n📈 特徴量重要度 TOP10:")
    print(importance.sort_values(ascending=False).head(10).to_string())


if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    train(conn)
    conn.close()
