#!/usr/bin/env python3
"""
株EV（期待値）計算モジュール
競馬のEV = 推定勝率 × オッズ - 1 を株に転用

株EV = P_up × 目標リターン - (1 - P_up) × 損切りライン
例）上昇確率60%、目標+15%、損切-5% → EV = 0.60×0.15 - 0.40×0.05 = 0.07 (+7%)
"""


def calc_stock_ev(
    p_up: float,
    target_return: float = 0.10,
    stop_loss: float = 0.05,
    ev_max: float = 0.50,
) -> dict:
    """
    p_up        : モデルが出した上昇確率（0〜1）
    target_return: 利確目標（デフォルト+10%）
    stop_loss   : 損切ライン（デフォルト-5%）
    """
    ev = p_up * target_return - (1 - p_up) * stop_loss
    ev = min(ev, ev_max)

    if ev >= 0.10:
        label = "★★★ 高期待値"
    elif ev >= 0.05:
        label = "★★  プラス期待値"
    elif ev >= 0.00:
        label = "★   妙味あり"
    elif ev >= -0.03:
        label = "    平均的"
    else:
        label = "    見送り推奨"

    return {
        "ev": round(ev, 4),
        "label": label,
        "p_up": round(p_up, 3),
        "target_return": target_return,
        "stop_loss": stop_loss,
    }


def kelly_bet_size(p_up: float, target_return: float, stop_loss: float,
                   fraction: float = 0.5, max_pct: float = 0.05) -> float:
    """
    ハーフケリー基準でのポジションサイズ（総資金に対する割合）
    fraction=0.5 はハーフケリー（競馬システムと同じ）
    """
    if target_return <= 0 or stop_loss <= 0:
        return 0.0
    b = target_return / stop_loss
    q = 1 - p_up
    kelly = (b * p_up - q) / b
    size = max(0.0, kelly * fraction)
    return min(size, max_pct)


def rank_candidates(records: list, min_ev: float = 0.0) -> list:
    """
    EVでフィルタ＆ランキング
    records: [{"symbol":..., "p_up":..., "close":..., ...}, ...]
    """
    results = []
    for r in records:
        p_up = r.get("p_up", 0.5)
        ev_result = calc_stock_ev(p_up)
        if ev_result["ev"] < min_ev:
            continue
        kelly = kelly_bet_size(p_up, ev_result["target_return"], ev_result["stop_loss"])
        results.append({
            **r,
            **ev_result,
            "kelly_pct": round(kelly * 100, 2),
        })
    return sorted(results, key=lambda x: x["ev"], reverse=True)


if __name__ == "__main__":
    samples = [
        {"symbol": "NVDA",   "p_up": 0.72, "close": 950.0},
        {"symbol": "7203.T", "p_up": 0.55, "close": 3800.0},
        {"symbol": "BTC-USD","p_up": 0.61, "close": 95000.0},
        {"symbol": "META",   "p_up": 0.40, "close": 510.0},
    ]
    ranked = rank_candidates(samples, min_ev=0.0)
    print(f"\n{'銘柄':<12} {'P_up':>6} {'EV':>7} {'kelly%':>7}  {'評価'}")
    print("-" * 55)
    for r in ranked:
        print(f"{r['symbol']:<12} {r['p_up']:>5.1%} {r['ev']:>+6.3f} {r['kelly_pct']:>6.1f}%  {r['label']}")
