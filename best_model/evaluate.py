# evaluate.py
"""
Evaluate the two-agent EGX30 pipeline (v2, TA-grounded) against actual moves.

Inputs:
  - A directory of prediction JSON files (outputs of pipeline3.py)
  - Daily-returns parquet      (returns_daily.parquet)
  - Daily-sector-returns parquet (sector_returns_daily.parquet)

For every prediction file, both horizons (short, medium) are scored:
  - actual cumulative abnormal return (AR) over the horizon window
  - directional hit / miss (with per-horizon neutral band)
  - magnitude   hit / miss (correct bucket AND correct direction)

Outputs (under --out):
  scorecard.csv      long format -- one row per (article, horizon)
  aggregate.json     hit rates by horizon / news_type / TA signal /
                     priced_in / confidence bucket

Usage:
  python evaluate.py --predictions-dir outputs/predictions_v2 \
                     --returns returns_daily.parquet \
                     --sector-returns sector_returns_daily.parquet \
                     --out outputs/evaluation_v2
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
#  Horizon configuration  (must mirror prompts.py)
# -----------------------------------------------------------------------------

HORIZONS = ["short", "medium"]

HORIZON_TRADING_DAYS = {
    "short":  2,    # t+1, t+2
    "medium": 20,   # t+1 .. t+20
}

HORIZON_NEUTRAL_BAND_PCT = {
    "short":  1.5,
    "medium": 4.0,
}

MAGNITUDE_BANDS_PCT = {
    "short":  {"small": (0.0, 1.5), "medium": (1.5, 4.0), "large": (4.0, float("inf"))},
    "medium": {"small": (0.0, 4.0), "medium": (4.0, 8.0), "large": (8.0, float("inf"))},
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
    Cumulative abnormal return over the horizon window (strictly forward from t+1).
    AR_t = stock_log_return_t - sector_log_return_t.
    Cumulative AR = sum of daily ARs over the window, expressed in percent.
    """
    if ticker not in returns.columns:
        return None
    sector = SECTOR_MAP.get(ticker)
    if sector is None or sector not in sector_returns.columns:
        return None

    future = returns.loc[returns.index > event_date]
    if future.empty:
        return None
    window = future.head(n_trading_days)
    if len(window) < n_trading_days:
        return None

    stock_window  = window[ticker].dropna()
    sector_window = sector_returns.loc[window.index, sector].dropna()
    common = stock_window.index.intersection(sector_window.index)
    if len(common) < n_trading_days:
        return None

    ar  = stock_window.loc[common] - sector_window.loc[common]
    cum = (np.exp(ar.sum()) - 1.0) * 100.0
    return float(cum)


def directional_hit(predicted_direction: str, actual_ar_pct: float, horizon: str) -> bool | None:
    band = HORIZON_NEUTRAL_BAND_PCT[horizon]
    if predicted_direction == "neutral":  return abs(actual_ar_pct) <= band
    if predicted_direction == "up":       return actual_ar_pct >  band
    if predicted_direction == "down":     return actual_ar_pct < -band
    return None


def magnitude_in_band(predicted_magnitude: str, actual_ar_pct: float, horizon: str) -> bool | None:
    bands = MAGNITUDE_BANDS_PCT[horizon]
    if predicted_magnitude not in bands:
        return None
    lo, hi = bands[predicted_magnitude]
    return lo <= abs(actual_ar_pct) < hi


def observed_magnitude(actual_ar_pct: float, horizon: str) -> str:
    abs_ar = abs(actual_ar_pct)
    for name, (lo, hi) in MAGNITUDE_BANDS_PCT[horizon].items():
        if lo <= abs_ar < hi:
            return name
    return "large"


def observed_direction(actual_ar_pct: float, horizon: str) -> str:
    band = HORIZON_NEUTRAL_BAND_PCT[horizon]
    if abs(actual_ar_pct) <= band:
        return "neutral"
    return "up" if actual_ar_pct > 0 else "down"


# -----------------------------------------------------------------------------
#  Scoring a single prediction file  ->  one row per horizon
# -----------------------------------------------------------------------------

def score_prediction(pred: dict, returns: pd.DataFrame, sector_returns: pd.DataFrame) -> list[dict]:
    primary        = pred["critic_agent"]["output"]["primary_company"]
    ticker         = primary["ticker"]
    news_type      = primary.get("news_type")
    already_priced = primary.get("already_priced_in")
    ta_signals     = primary.get("ta_signals_cited") or []
    per_horizon    = primary.get("per_horizon", {}) or {}

    event_date_str = pred["article_date"]
    event_date     = pd.Timestamp(event_date_str)

    rows: list[dict] = []
    for h in HORIZONS:
        h_pred = per_horizon.get(h) or {}
        direction  = h_pred.get("direction")
        magnitude  = h_pred.get("magnitude")
        confidence = h_pred.get("confidence")

        n_days    = HORIZON_TRADING_DAYS[h]
        actual_ar = cumulative_ar_pct(returns, sector_returns, ticker, event_date, n_days)

        if actual_ar is None or direction is None or magnitude is None:
            rows.append({
                "ticker":             ticker,
                "event_date":         event_date_str,
                "horizon":            h,
                "news_type":          news_type,
                "already_priced_in":  already_priced,
                "ta_signals_cited":   "|".join(ta_signals) if ta_signals else "",
                "predicted_direction": direction,
                "predicted_magnitude": magnitude,
                "confidence":          confidence,
                "actual_ar_pct":       None if actual_ar is None else round(actual_ar, 3),
                "observed_direction":  None,
                "observed_magnitude":  None,
                "directional_hit":     None,
                "magnitude_hit":       None,
                "status": "no_data" if actual_ar is None else "no_prediction",
            })
            continue

        dir_hit  = directional_hit(direction, actual_ar, h)
        mag_band = magnitude_in_band(magnitude, actual_ar, h)
        mag_hit  = bool(dir_hit) and bool(mag_band) if mag_band is not None else None

        rows.append({
            "ticker":              ticker,
            "event_date":          event_date_str,
            "horizon":             h,
            "news_type":           news_type,
            "already_priced_in":   already_priced,
            "ta_signals_cited":    "|".join(ta_signals) if ta_signals else "",
            "predicted_direction": direction,
            "predicted_magnitude": magnitude,
            "confidence":          confidence,
            "actual_ar_pct":       round(actual_ar, 3),
            "observed_direction":  observed_direction(actual_ar, h),
            "observed_magnitude":  observed_magnitude(actual_ar, h),
            "directional_hit":     bool(dir_hit),
            "magnitude_hit":       mag_hit,
            "status":              "scored",
        })

    return rows


# -----------------------------------------------------------------------------
#  Aggregation
# -----------------------------------------------------------------------------

def _hit_rate(records: list[dict], key: str) -> dict:
    buckets: dict = defaultdict(lambda: {"n": 0, "dir_hits": 0, "mag_hits": 0})
    for r in records:
        if r["status"] != "scored":
            continue
        k = r.get(key)
        if k is None:
            continue
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


def _hit_rate_by_two(records: list[dict], k1: str, k2: str) -> dict:
    buckets: dict = defaultdict(lambda: {"n": 0, "dir_hits": 0, "mag_hits": 0})
    for r in records:
        if r["status"] != "scored":
            continue
        a, b = r.get(k1), r.get(k2)
        if a is None or b is None:
            continue
        buckets[(a, b)]["n"] += 1
        if r["directional_hit"]:
            buckets[(a, b)]["dir_hits"] += 1
        if r["magnitude_hit"]:
            buckets[(a, b)]["mag_hits"] += 1

    out: dict = {}
    for (a, b), v in buckets.items():
        n = v["n"]
        out.setdefault(a, {})[b] = {
            "n": n,
            "directional_hit_rate": round(v["dir_hits"] / n, 3) if n else None,
            "magnitude_hit_rate":   round(v["mag_hits"] / n, 3) if n else None,
        }
    return out


def _confidence_bucket(c: float | None) -> str | None:
    if c is None:           return None
    if c < 0.4:             return "0.0-0.4"
    if c < 0.6:             return "0.4-0.6"
    if c < 0.8:             return "0.6-0.8"
    return "0.8-1.0"


def _confidence_aggregation(records: list[dict]) -> dict:
    """Confidence-bucket hit rates, per horizon."""
    buckets: dict = defaultdict(lambda: {"n": 0, "dir_hits": 0, "mag_hits": 0})
    for r in records:
        if r["status"] != "scored":
            continue
        cb = _confidence_bucket(r.get("confidence"))
        if cb is None:
            continue
        key = (r["horizon"], cb)
        buckets[key]["n"] += 1
        if r["directional_hit"]:
            buckets[key]["dir_hits"] += 1
        if r["magnitude_hit"]:
            buckets[key]["mag_hits"] += 1

    out: dict = {}
    for (h, cb), v in buckets.items():
        n = v["n"]
        out.setdefault(h, {})[cb] = {
            "n": n,
            "directional_hit_rate": round(v["dir_hits"] / n, 3) if n else None,
            "magnitude_hit_rate":   round(v["mag_hits"] / n, 3) if n else None,
        }
    return out


def _ta_signal_hit_rate(records: list[dict]) -> dict:
    """
    Per TA signal: hit rate when that signal was cited. Helps identify which
    signals correlate with accurate predictions and which are noise.
    """
    buckets: dict = defaultdict(lambda: {"n": 0, "dir_hits": 0, "mag_hits": 0})
    for r in records:
        if r["status"] != "scored":
            continue
        signals = (r.get("ta_signals_cited") or "").split("|")
        for sig in signals:
            sig = sig.strip()
            if not sig:
                continue
            buckets[sig]["n"] += 1
            if r["directional_hit"]:
                buckets[sig]["dir_hits"] += 1
            if r["magnitude_hit"]:
                buckets[sig]["mag_hits"] += 1

    out: dict = {}
    for sig, v in buckets.items():
        n = v["n"]
        out[sig] = {
            "n": n,
            "directional_hit_rate": round(v["dir_hits"] / n, 3) if n else None,
            "magnitude_hit_rate":   round(v["mag_hits"] / n, 3) if n else None,
        }
    return out


# -----------------------------------------------------------------------------
#  Top-level aggregate
# -----------------------------------------------------------------------------

def aggregate(records: list[dict]) -> dict:
    scored  = [r for r in records if r["status"] == "scored"]
    no_data = sum(1 for r in records if r["status"] == "no_data")
    no_pred = sum(1 for r in records if r["status"] == "no_prediction")

    overall_dir = (
        sum(1 for r in scored if r["directional_hit"]) / len(scored)
        if scored else None
    )
    overall_mag = (
        sum(1 for r in scored if r["magnitude_hit"]) / len(scored)
        if scored else None
    )

    return {
        "n_total_rows":                 len(records),
        "n_scored":                     len(scored),
        "n_no_data":                    no_data,
        "n_no_prediction":              no_pred,
        "overall_directional_hit_rate": round(overall_dir, 3) if overall_dir is not None else None,
        "overall_magnitude_hit_rate":   round(overall_mag, 3) if overall_mag is not None else None,
        "by_horizon":                   _hit_rate(scored, "horizon"),
        "by_news_type":                 _hit_rate(scored, "news_type"),
        "by_direction":                 _hit_rate(scored, "predicted_direction"),
        "by_magnitude":                 _hit_rate(scored, "predicted_magnitude"),
        "by_already_priced_in":         _hit_rate(scored, "already_priced_in"),
        "horizon_x_news_type":          _hit_rate_by_two(scored, "horizon", "news_type"),
        "by_confidence_per_horizon":    _confidence_aggregation(scored),
        "by_ta_signal":                 _ta_signal_hit_rate(scored),
    }


# -----------------------------------------------------------------------------
#  CLI
# -----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions-dir", type=Path, default=Path("outputs/predictions_v2"))
    p.add_argument("--returns",         type=Path, default=Path("returns_daily.parquet"))
    p.add_argument("--sector-returns",  type=Path, default=Path("sector_returns_daily.parquet"))
    p.add_argument("--out",             type=Path, default=Path("outputs/evaluation_v2"))
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    returns = pd.read_parquet(args.returns)
    returns.index = pd.to_datetime(returns.index)
    sector_returns = pd.read_parquet(args.sector_returns)
    sector_returns.index = pd.to_datetime(sector_returns.index)

    records: list[dict] = []
    n_files = 0
    for pred_file in sorted(args.predictions_dir.glob("*.json")):
        n_files += 1
        with open(pred_file, "r", encoding="utf-8") as f:
            pred = json.load(f)
        try:
            rows = score_prediction(pred, returns, sector_returns)
            for r in rows:
                r["file"] = pred_file.name
            records.extend(rows)
        except Exception as e:
            print(f"[WARN] failed on {pred_file.name}: {e}")

    # ── Per-(article, horizon) scorecard, long format ─────────────────────────
    df = pd.DataFrame(records)
    df.to_csv(args.out / "scorecard.csv", index=False)

    # ── Aggregated metrics ────────────────────────────────────────────────────
    agg = aggregate(records)
    with open(args.out / "aggregate.json", "w", encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False, indent=2, default=str)

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\nScored {agg['n_scored']}/{agg['n_total_rows']} (article x horizon) rows "
          f"from {n_files} prediction file(s).")
    print(f"  no_data       : {agg['n_no_data']}    (missing future prices)")
    print(f"  no_prediction : {agg['n_no_prediction']}  (LLM returned no per-horizon entry)")
    print(f"\nOverall directional hit rate : {agg['overall_directional_hit_rate']}")
    print(f"Overall magnitude   hit rate : {agg['overall_magnitude_hit_rate']}")

    print("\nBy horizon:")
    for h in HORIZONS:
        row = agg["by_horizon"].get(h, {})
        print(f"  {h:<7}  n={row.get('n', 0):<4}  "
              f"dir={row.get('directional_hit_rate')}  "
              f"mag={row.get('magnitude_hit_rate')}")

    if agg["by_ta_signal"]:
        print("\nTop TA signals by directional hit rate (n>=3):")
        ranked = sorted(
            ((sig, m) for sig, m in agg["by_ta_signal"].items() if m["n"] >= 3),
            key=lambda x: (x[1]["directional_hit_rate"] or 0),
            reverse=True,
        )
        for sig, m in ranked[:10]:
            print(f"  {sig:<28}  n={m['n']:<3}  dir={m['directional_hit_rate']}  mag={m['magnitude_hit_rate']}")

    print(f"\nWrote: {args.out / 'scorecard.csv'}")
    print(f"Wrote: {args.out / 'aggregate.json'}")


if __name__ == "__main__":
    main()
