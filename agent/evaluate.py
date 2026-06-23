"""
Dual-target evaluator for the EGX News-Impact Agent (v2).

Scores THREE targets per (article, horizon):
  1. ticker_direction  — predicted vs sign(actual_ticker_return)
  2. sector_direction  — predicted vs sign(actual_sector_return)
  3. implied_abnormal  — derived from the two predictions vs
                         sign(actual_ticker_return - actual_sector_excl_target_return)

Baselines reported per target:
  always_up        — always predict UP
  market_direction — predict EGX30 / sector direction that day
  sentiment_sign   — use CAMeLBERT label as direction

Usage:
  python -m agent.evaluate \
    --predictions-dir agent/outputs/predictions \
    --returns         returns_daily.parquet \
    --sector-returns  sector_returns_daily.parquet \
    --out             agent/outputs/evaluation
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ACTIVE_HORIZONS, HORIZON_DAYS

# ---------------------------------------------------------------------------
# Sector map (same as parent evaluate.py)
# ---------------------------------------------------------------------------

SECTOR_MAP: dict[str, str] = {
    "ABUK": "Petroleum & Petrochemicals", "VLMRA": "Petroleum & Petrochemicals",
    "VLMR": "Petroleum & Petrochemicals", "SKPC": "Petroleum & Petrochemicals",
    "AMOC": "Petroleum & Petrochemicals", "MFPC": "Petroleum & Petrochemicals",
    "COMI": "Banking & Financial Services", "ADIB": "Banking & Financial Services",
    "CIEB": "Banking & Financial Services", "BTFH": "Banking & Financial Services",
    "HRHO": "Non-Bank Financials", "CCAP": "Non-Bank Financials",
    "FWRY": "Technology & Fintech", "EFIH": "Technology & Fintech",
    "RAYA": "Technology & Fintech",
    "PHDC": "Real Estate", "MASR": "Real Estate", "TMGH": "Real Estate",
    "EMFD": "Real Estate", "ORHD": "Real Estate",
    "ARCC": "Construction & Building Materials",
    "MCQE": "Construction & Building Materials",
    "ORAS": "Construction & Building Materials",
    "RMDA": "Healthcare & Pharmaceuticals", "ISPH": "Healthcare & Pharmaceuticals",
    "JUFO": "Food & Beverages",
    "EGAL": "Metals & Mining",
    "ETEL": "Telecommunications",
    "ORWE": "Textiles",
    "EAST": "Tobacco & Consumer Goods",
    "GBCO": "Diversified Holding",
}


# ---------------------------------------------------------------------------
# Return helpers
# ---------------------------------------------------------------------------

def _cumulative_return_pct(
    returns: pd.DataFrame,
    col: str,
    event_date: pd.Timestamp,
    n_days: int,
    inclusive: bool,
) -> tuple[float | None, pd.DatetimeIndex | None]:
    if col not in returns.columns:
        return None, None
    future = returns.loc[returns.index >= event_date] if inclusive else returns.loc[returns.index > event_date]
    window = future.head(n_days)
    if len(window) < n_days:
        return None, None
    series = window[col].dropna()
    if len(series) < n_days:
        return None, None
    cum = (np.exp(series.sum()) - 1.0) * 100.0
    return float(cum), series.index


def _neutral_band(atr: float | None, horizon: str) -> float:
    n = HORIZON_DAYS[horizon]
    if atr and atr > 0:
        return round(0.5 * atr * math.sqrt(n), 2)
    return {"short": 1.0, "medium": 2.5}[horizon]


def _observed_direction(ret_pct: float, band: float) -> str:
    if abs(ret_pct) <= band:
        return "neutral"
    return "up" if ret_pct > 0 else "down"


def _directional_hit(predicted: str, actual_ret: float, band: float) -> bool | None:
    if predicted == "neutral":
        return abs(actual_ret) <= band
    if predicted == "up":
        return actual_ret > band
    if predicted == "down":
        return actual_ret < -band
    return None


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def _baseline_always_up(_: float) -> str:
    return "up"


def _baseline_sentiment(sentiment_label: str) -> str:
    return {"positive": "up", "negative": "down"}.get(sentiment_label, "neutral")


def _baseline_market(market_ret: float | None, band: float) -> str:
    if market_ret is None:
        return "neutral"
    return _observed_direction(market_ret, band)


# ---------------------------------------------------------------------------
# Score one prediction file
# ---------------------------------------------------------------------------

def score_prediction(
    pred: dict,
    ticker_returns: pd.DataFrame,
    sector_returns: pd.DataFrame,
) -> list[dict]:
    ticker      = pred["ticker"]
    event_date  = pd.Timestamp(pred["event_date"])
    inclusive   = pred.get("in_trading_hours", False)
    sector_name = SECTOR_MAP.get(ticker)

    # ATR proxies stored in the prediction trace (from context)
    ticker_atr  = (pred.get("context") or {}).get("ticker_atr")
    sector_atr  = (pred.get("context") or {}).get("sector_atr")

    sentiment_label = pred.get("sentiment_label", "neutral")

    rows: list[dict] = []
    for h in ACTIVE_HORIZONS:
        n_days = HORIZON_DAYS[h]
        h_pred = (pred.get("outlook") or {}).get(h) or {}

        pred_ticker  = (h_pred.get("ticker") or {}).get("direction")
        pred_sector  = (h_pred.get("sector") or {}).get("direction")
        pred_implied = h_pred.get("implied_abnormal")
        conf_ticker  = (h_pred.get("ticker") or {}).get("confidence")
        conf_sector  = (h_pred.get("sector") or {}).get("confidence")

        # --- Realised returns ---
        t_ret, _ = _cumulative_return_pct(ticker_returns, ticker, event_date, n_days, inclusive)
        s_ret, _ = (
            _cumulative_return_pct(sector_returns, sector_name, event_date, n_days, inclusive)
            if sector_name else (None, None)
        )

        # Sector excl-target for implied_abnormal ground truth
        col_excl = f"{sector_name}_excl_{ticker}"
        se_ret, _ = (
            _cumulative_return_pct(sector_returns, col_excl, event_date, n_days, inclusive)
            if col_excl in sector_returns.columns else (None, None)
        )
        abnormal_ret = (
            round(t_ret - se_ret, 3)
            if t_ret is not None and se_ret is not None else None
        )

        t_band = _neutral_band(ticker_atr, h)
        s_band = _neutral_band(sector_atr, h)

        # Baseline: market direction == sector direction for that horizon
        market_dir = _observed_direction(s_ret, s_band) if s_ret is not None else None

        base: dict = {
            "ticker":          ticker,
            "sector":          sector_name,
            "event_date":      str(pred["event_date"]),
            "horizon":         h,
            "event_type":      pred.get("event_type"),
            "critic_invoked":  pred.get("critic_invoked", False),
            # predictions
            "pred_ticker":     pred_ticker,
            "pred_sector":     pred_sector,
            "pred_implied":    pred_implied,
            "conf_ticker":     conf_ticker,
            "conf_sector":     conf_sector,
            # realised
            "actual_ticker_ret":   None if t_ret is None else round(t_ret, 3),
            "actual_sector_ret":   None if s_ret is None else round(s_ret, 3),
            "actual_abnormal_ret": abnormal_ret,
        }

        if t_ret is None:
            rows.append({**base, "status": "no_data"})
            continue

        obs_ticker  = _observed_direction(t_ret, t_band)
        obs_sector  = _observed_direction(s_ret, s_band) if s_ret is not None else None
        obs_implied = _observed_direction(abnormal_ret, t_band) if abnormal_ret is not None else None

        hit_ticker  = _directional_hit(pred_ticker, t_ret, t_band) if pred_ticker else None
        hit_sector  = _directional_hit(pred_sector, s_ret, s_band) if (pred_sector and s_ret is not None) else None
        hit_implied = _directional_hit(pred_implied, abnormal_ret, t_band) if (pred_implied and abnormal_ret is not None) else None

        # Baselines
        bl_always_up_t  = _directional_hit("up", t_ret, t_band)
        bl_always_up_s  = _directional_hit("up", s_ret, s_band) if s_ret is not None else None
        bl_market_t     = _directional_hit(market_dir, t_ret, t_band) if market_dir else None
        bl_market_s     = _directional_hit(market_dir, s_ret, s_band) if (market_dir and s_ret is not None) else None
        bl_sent_t       = _directional_hit(_baseline_sentiment(sentiment_label), t_ret, t_band)
        bl_sent_s       = _directional_hit(_baseline_sentiment(sentiment_label), s_ret, s_band) if s_ret is not None else None

        rows.append({
            **base,
            "status":           "scored",
            "obs_ticker":       obs_ticker,
            "obs_sector":       obs_sector,
            "obs_implied":      obs_implied,
            "hit_ticker":       bool(hit_ticker)  if hit_ticker  is not None else None,
            "hit_sector":       bool(hit_sector)  if hit_sector  is not None else None,
            "hit_implied":      bool(hit_implied) if hit_implied is not None else None,
            # baselines
            "bl_always_up_ticker":    bool(bl_always_up_t) if bl_always_up_t is not None else None,
            "bl_always_up_sector":    bool(bl_always_up_s) if bl_always_up_s is not None else None,
            "bl_market_ticker":       bool(bl_market_t)    if bl_market_t    is not None else None,
            "bl_market_sector":       bool(bl_market_s)    if bl_market_s    is not None else None,
            "bl_sentiment_ticker":    bool(bl_sent_t)      if bl_sent_t      is not None else None,
            "bl_sentiment_sector":    bool(bl_sent_s)      if bl_sent_s      is not None else None,
        })

    return rows


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def _hit_rate(records: list[dict], hit_col: str) -> dict:
    total = sum(1 for r in records if r.get(hit_col) is not None)
    hits  = sum(1 for r in records if r.get(hit_col) is True)
    return {"n": total, "hit_rate": round(hits / total, 3) if total else None}


def aggregate(records: list[dict]) -> dict:
    scored = [r for r in records if r.get("status") == "scored"]

    def _by(key: str, hit_col: str) -> dict:
        buckets: dict = defaultdict(list)
        for r in scored:
            k = r.get(key)
            if k is not None:
                buckets[k].append(r)
        return {k: _hit_rate(v, hit_col) for k, v in buckets.items()}

    return {
        "n_total":   len(records),
        "n_scored":  len(scored),
        "n_no_data": sum(1 for r in records if r.get("status") == "no_data"),
        "ticker": {
            "overall":        _hit_rate(scored, "hit_ticker"),
            "baseline_always_up": _hit_rate(scored, "bl_always_up_ticker"),
            "baseline_market":    _hit_rate(scored, "bl_market_ticker"),
            "baseline_sentiment": _hit_rate(scored, "bl_sentiment_ticker"),
            "by_horizon":     _by("horizon",    "hit_ticker"),
            "by_event_type":  _by("event_type", "hit_ticker"),
            "by_sector":      _by("sector",     "hit_ticker"),
        },
        "sector": {
            "overall":        _hit_rate(scored, "hit_sector"),
            "baseline_always_up": _hit_rate(scored, "bl_always_up_sector"),
            "baseline_market":    _hit_rate(scored, "bl_market_sector"),
            "baseline_sentiment": _hit_rate(scored, "bl_sentiment_sector"),
            "by_horizon":     _by("horizon",    "hit_sector"),
            "by_event_type":  _by("event_type", "hit_sector"),
            "by_sector":      _by("sector",     "hit_sector"),
        },
        "implied_abnormal": {
            "overall":       _hit_rate(scored, "hit_implied"),
            "by_horizon":    _by("horizon",    "hit_implied"),
            "by_event_type": _by("event_type", "hit_implied"),
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Dual-target evaluation for EGX agent v2")
    p.add_argument("--predictions-dir", type=Path, default=Path("agent/outputs/predictions"))
    p.add_argument("--returns",         type=Path, default=Path("returns_daily.parquet"))
    p.add_argument("--sector-returns",  type=Path, default=Path("sector_returns_daily.parquet"))
    p.add_argument("--out",             type=Path, default=Path("agent/outputs/evaluation"))
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    ticker_returns = pd.read_parquet(args.returns)
    ticker_returns.index = pd.to_datetime(ticker_returns.index)

    sector_returns = pd.read_parquet(args.sector_returns)
    sector_returns.index = pd.to_datetime(sector_returns.index)

    all_rows: list[dict] = []
    n_files = 0
    for pred_file in sorted(args.predictions_dir.glob("*.json")):
        n_files += 1
        with open(pred_file, encoding="utf-8") as f:
            pred = json.load(f)
        try:
            rows = score_prediction(pred, ticker_returns, sector_returns)
            for r in rows:
                r["file"] = pred_file.name
            all_rows.extend(rows)
        except Exception as exc:
            print(f"[WARN] {pred_file.name}: {exc}")

    df = pd.DataFrame(all_rows)
    df.to_csv(args.out / "scorecard.csv", index=False)

    agg = aggregate(all_rows)
    with open(args.out / "aggregate.json", "w", encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False, indent=2, default=str)

    # Print summary
    print(f"\nEvaluated {agg['n_scored']}/{agg['n_total']} rows from {n_files} file(s).")
    for target in ("ticker", "sector", "implied_abnormal"):
        t = agg[target]
        ov = t["overall"]
        print(f"\n{target.upper()} overall: {ov['hit_rate']} (n={ov['n']})")
        if "baseline_always_up" in t:
            print(f"  baseline always_up : {t['baseline_always_up']['hit_rate']}")
            print(f"  baseline market    : {t['baseline_market']['hit_rate']}")
            print(f"  baseline sentiment : {t['baseline_sentiment']['hit_rate']}")
        if "by_horizon" in t:
            for h, m in t["by_horizon"].items():
                print(f"  {h:<8}: {m['hit_rate']} (n={m['n']})")

    print(f"\nWrote: {args.out / 'scorecard.csv'}")
    print(f"Wrote: {args.out / 'aggregate.json'}")


if __name__ == "__main__":
    main()
