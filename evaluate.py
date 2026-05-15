# evaluate.py
"""
Evaluate the multi-agent EGX30 pipeline against actual price moves.

Given:
  - A directory of prediction JSON files (outputs of pipeline.py)
  - A daily-returns parquet file (returns_daily.parquet)
  - A daily-sector-returns parquet file (sector_returns_daily.parquet)
  - sector_map (hardcoded)

Computes:
  - Per-prediction: actual cumulative AR over the predicted horizon,
                    directional hit (yes/no), magnitude hit (yes/no)
  - Aggregated: hit ratios by horizon, by news_type, by magnitude,
                by confidence bucket, and overall

Usage:
  python evaluate.py --predictions-dir outputs/predictions \
                     --returns returns_daily.parquet \
                     --sector-returns sector_returns_daily.parquet \
                     --out outputs/evaluation
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
#  Locked horizon configuration (mirror of prompts.py)
# -----------------------------------------------------------------------------

HORIZON_TRADING_DAYS = {
    "1d":   1,
    "2-5d": 5,
    "1-2w": 10,
    "2-4w": 20,
    "1-3m": 60,
}

HORIZON_NEUTRAL_BAND_PCT = {
    "1d":   1.0,
    "2-5d": 2.0,
    "1-2w": 3.0,
    "2-4w": 4.0,
    "1-3m": 5.0,
}

# Magnitude thresholds (% AR) -- mid-point of each class
MAGNITUDE_BANDS_PCT = {
    "small":  (0.0,  1.5),
    "medium": (1.5,  3.5),
    "large":  (3.5,  100.0),
}

SECTOR_MAP = {
    "ABUK": "Petroleum & Petrochemicals", "SKPC": "Petroleum & Petrochemicals",
    "AMOC": "Petroleum & Petrochemicals", "MFPC": "Petroleum & Petrochemicals",
    "EKHO": "Petroleum & Petrochemicals",
    "COMI": "Banking", "ADIB": "Banking", "CIEB": "Banking",
    "HRHO": "Non-Bank Financials", "CCAP": "Non-Bank Financials",
    "BTFH": "Non-Bank Financials",
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


# -----------------------------------------------------------------------------
#  Core computation
# -----------------------------------------------------------------------------

def cumulative_ar_pct(
    returns: pd.DataFrame,
    sector_returns: pd.DataFrame,
    ticker: str,
    event_date: pd.Timestamp,
    n_trading_days: int,
) -> float | None:
    """
    Cumulative abnormal return from t+1 close to t+n close, in percent.
    AR_t = stock_log_return_t - sector_log_return_t
    Cumulative AR = sum of daily ARs over the window.
    """
    if ticker not in returns.columns:
        return None
    sector = SECTOR_MAP.get(ticker)
    if sector is None or sector not in sector_returns.columns:
        return None

    # Find the first trading day strictly AFTER event_date
    future = returns.loc[returns.index > event_date]
    if future.empty:
        return None
    window = future.head(n_trading_days)
    if len(window) < n_trading_days:
        # Not enough future data yet -- evaluation cannot complete
        return None

    stock_window  = window[ticker].dropna()
    sector_window = sector_returns.loc[window.index, sector].dropna()
    common = stock_window.index.intersection(sector_window.index)
    if len(common) < n_trading_days:
        return None

    ar = stock_window.loc[common] - sector_window.loc[common]
    # log returns -> sum then exp-1 -> percent
    cum = (np.exp(ar.sum()) - 1.0) * 100.0
    return float(cum)


def directional_hit(
    predicted_direction: str,
    actual_ar_pct: float,
    horizon: str,
) -> bool | None:
    band = HORIZON_NEUTRAL_BAND_PCT[horizon]
    if predicted_direction == "neutral":
        return abs(actual_ar_pct) <= band
    if predicted_direction == "up":
        return actual_ar_pct > band
    if predicted_direction == "down":
        return actual_ar_pct < -band
    return None


def magnitude_hit(
    predicted_magnitude: str,
    predicted_direction: str,
    actual_ar_pct: float,
) -> bool | None:
    """
    Magnitude hit = direction correct AND |actual AR| falls in the
    predicted magnitude band.
    """
    if predicted_magnitude not in MAGNITUDE_BANDS_PCT:
        return None
    lo, hi = MAGNITUDE_BANDS_PCT[predicted_magnitude]
    abs_ar = abs(actual_ar_pct)
    in_band = lo <= abs_ar < hi
    if predicted_direction == "neutral":
        # For neutral, magnitude is implicitly small; treat as hit
        # only if direction-hit is also true (handled by caller).
        return in_band
    return in_band


# -----------------------------------------------------------------------------
#  Scoring a single prediction file
# -----------------------------------------------------------------------------

def score_prediction(
    pred: dict,
    returns: pd.DataFrame,
    sector_returns: pd.DataFrame,
) -> dict:
    primary = pred["critic_agent"]["output"]["primary_company"]
    ticker  = primary["ticker"]
    horizon = primary["horizon"]
    direction = primary["direction"]
    magnitude = primary["magnitude"]
    news_type = primary["news_type"]
    confidence = primary["confidence"]

    event_date_str = pred["article_date"]
    event_date = pd.Timestamp(event_date_str)

    n_days = HORIZON_TRADING_DAYS[horizon]
    actual_ar = cumulative_ar_pct(returns, sector_returns, ticker, event_date, n_days)

    if actual_ar is None:
        return {
            "ticker": ticker,
            "event_date": event_date_str,
            "horizon": horizon,
            "news_type": news_type,
            "predicted_direction": direction,
            "predicted_magnitude": magnitude,
            "confidence": confidence,
            "actual_ar_pct": None,
            "directional_hit": None,
            "magnitude_hit": None,
            "status": "no_data",
        }

    dir_hit = directional_hit(direction, actual_ar, horizon)
    mag_in_band = magnitude_hit(magnitude, direction, actual_ar)
    mag_hit = bool(dir_hit) and bool(mag_in_band) if mag_in_band is not None else None

    return {
        "ticker": ticker,
        "event_date": event_date_str,
        "horizon": horizon,
        "news_type": news_type,
        "predicted_direction": direction,
        "predicted_magnitude": magnitude,
        "confidence": confidence,
        "actual_ar_pct": round(actual_ar, 3),
        "directional_hit": bool(dir_hit),
        "magnitude_hit": mag_hit,
        "status": "scored",
    }


# -----------------------------------------------------------------------------
#  Aggregation
# -----------------------------------------------------------------------------

def _hit_rate(records: list[dict], key: str) -> dict:
    """Compute hit rate stratified by `key` (e.g. 'horizon', 'news_type')."""
    buckets = defaultdict(lambda: {"n": 0, "dir_hits": 0, "mag_hits": 0})
    for r in records:
        if r["status"] != "scored":
            continue
        k = r[key]
        buckets[k]["n"] += 1
        if r["directional_hit"]:
            buckets[k]["dir_hits"] += 1
        if r["magnitude_hit"]:
            buckets[k]["mag_hits"] += 1
    out = {}
    for k, b in buckets.items():
        n = b["n"]
        out[k] = {
            "n": n,
            "directional_hit_rate": round(b["dir_hits"] / n, 3) if n else None,
            "magnitude_hit_rate":   round(b["mag_hits"] / n, 3) if n else None,
        }
    return out


def _confidence_bucket(c: float) -> str:
    if c < 0.4:  return "0.0-0.4"
    if c < 0.6:  return "0.4-0.6"
    if c < 0.8:  return "0.6-0.8"
    return "0.8-1.0"


def aggregate(records: list[dict]) -> dict:
    scored = [r for r in records if r["status"] == "scored"]
    no_data = sum(1 for r in records if r["status"] == "no_data")

    overall_dir = sum(1 for r in scored if r["directional_hit"]) / len(scored) if scored else None
    overall_mag = sum(1 for r in scored if r["magnitude_hit"])   / len(scored) if scored else None

    by_conf = defaultdict(lambda: {"n": 0, "dir_hits": 0, "mag_hits": 0})
    for r in scored:
        b = _confidence_bucket(r["confidence"])
        by_conf[b]["n"] += 1
        if r["directional_hit"]: by_conf[b]["dir_hits"] += 1
        if r["magnitude_hit"]:   by_conf[b]["mag_hits"] += 1
    by_conf_out = {
        b: {
            "n": v["n"],
            "directional_hit_rate": round(v["dir_hits"] / v["n"], 3) if v["n"] else None,
            "magnitude_hit_rate":   round(v["mag_hits"] / v["n"], 3) if v["n"] else None,
        }
        for b, v in by_conf.items()
    }

    return {
        "n_total":             len(records),
        "n_scored":            len(scored),
        "n_no_data":           no_data,
        "overall_directional_hit_rate": round(overall_dir, 3) if overall_dir is not None else None,
        "overall_magnitude_hit_rate":   round(overall_mag, 3) if overall_mag is not None else None,
        "by_horizon":     _hit_rate(scored, "horizon"),
        "by_news_type":   _hit_rate(scored, "news_type"),
        "by_direction":   _hit_rate(scored, "predicted_direction"),
        "by_magnitude":   _hit_rate(scored, "predicted_magnitude"),
        "by_confidence":  by_conf_out,
    }


# -----------------------------------------------------------------------------
#  CLI
# -----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions-dir", type=Path, default=Path("outputs/predictions"))
    p.add_argument("--returns",         type=Path, default=Path("returns_daily.parquet"))
    p.add_argument("--sector-returns",  type=Path, default=Path("sector_returns_daily.parquet"))
    p.add_argument("--out",             type=Path, default=Path("outputs/evaluation"))
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    returns = pd.read_parquet(args.returns)
    returns.index = pd.to_datetime(returns.index)
    sector_returns = pd.read_parquet(args.sector_returns)
    sector_returns.index = pd.to_datetime(sector_returns.index)

    records = []
    for pred_file in sorted(args.predictions_dir.glob("*.json")):
        with open(pred_file, "r", encoding="utf-8") as f:
            pred = json.load(f)
        try:
            rec = score_prediction(pred, returns, sector_returns)
            rec["file"] = pred_file.name
            records.append(rec)
        except Exception as e:
            print(f"[WARN] failed on {pred_file.name}: {e}")

    # Per-prediction scorecard
    df = pd.DataFrame(records)
    df.to_csv(args.out / "scorecard.csv", index=False)

    # Aggregated metrics
    agg = aggregate(records)
    with open(args.out / "aggregate.json", "w", encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)

    print(f"Scored {agg['n_scored']}/{agg['n_total']} predictions "
          f"({agg['n_no_data']} skipped for missing future data).")
    print(f"Overall directional hit rate: {agg['overall_directional_hit_rate']}")
    print(f"Overall magnitude  hit rate: {agg['overall_magnitude_hit_rate']}")
    print(f"Wrote: {args.out/'scorecard.csv'}")
    print(f"Wrote: {args.out/'aggregate.json'}")


if __name__ == "__main__":
    main()