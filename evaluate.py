# evaluate.py
"""
Evaluate the two-agent EGX30 pipeline (v2.1, TA-grounded, per-company bands).

Inputs:
  - A directory of prediction JSON files (outputs of pipeline3.py)
  - Daily-returns parquet  (returns_daily.parquet)

For every prediction file, all three horizons (short, medium, large) are scored:
  - actual cumulative return of the company over the horizon window
  - directional hit (predicted matches sign vs per-company neutral band)
  - magnitude   hit (correct bucket under the per-company bands AND direction right)

Window-start rule (mirrors pipeline3.py / features.news_in_trading_hours):
  news arrived during EGX hours (10:00-14:30) -> window starts AT event_date
  news arrived outside hours                  -> window starts at next trading day

Per-company bands: read from pred["context"]["dynamic_bands"], guaranteeing the
SAME bands the agent used at prediction time. Falls back to static bands if
the trace was produced by an older pipeline version.

Dividend adjustment: when news_type=dividend and the trace carries
dividend_info.yield_pct + ex_div_date inside the horizon window, the
mechanical ex-div drop is added BACK to the return before scoring.

Outputs (under --out):
  scorecard.csv      long format -- one row per (article, horizon)
  aggregate.json     hit rates by horizon / news_type / TA signal / confidence

Usage:
  python evaluate.py --predictions-dir outputs/predictions_v2 \
                     --returns returns_daily.parquet \
                     --out outputs/evaluation_v2
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
#  Horizon configuration -- imported from features.py (single source of truth).
#  To add/remove/resize horizons, edit features.HORIZON_TRADING_DAYS.
# -----------------------------------------------------------------------------

from features import HORIZON_TRADING_DAYS

HORIZONS = list(HORIZON_TRADING_DAYS.keys())

# Ticker -> sector mapping (EGX30 universe)
SECTOR_MAP: dict[str, str] = {
    # Petroleum & Petrochemicals
    "ABUK":  "Petroleum & Petrochemicals",
    "VLMRA": "Petroleum & Petrochemicals",
    "VLMR":  "Petroleum & Petrochemicals",
    "SKPC":  "Petroleum & Petrochemicals",
    "AMOC":  "Petroleum & Petrochemicals",
    "MFPC":  "Petroleum & Petrochemicals",
    # Banking & Financial Services
    "COMI":  "Banking & Financial Services",
    "ADIB":  "Banking & Financial Services",
    "CIEB":  "Banking & Financial Services",
    "BTFH":  "Banking & Financial Services",
    # Non-Bank Financials
    "HRHO":  "Non-Bank Financials",
    "CCAP":  "Non-Bank Financials",
    # Technology & Fintech
    "FWRY":  "Technology & Fintech",
    "EFIH":  "Technology & Fintech",
    "RAYA":  "Technology & Fintech",
    # Real Estate
    "PHDC":  "Real Estate",
    "MASR":  "Real Estate",
    "TMGH":  "Real Estate",
    "EMFD":  "Real Estate",
    "ORHD":  "Real Estate",
    # Construction & Building Materials
    "ARCC":  "Construction & Building Materials",
    "MCQE":  "Construction & Building Materials",
    "ORAS":  "Construction & Building Materials",
    # Healthcare & Pharmaceuticals
    "RMDA":  "Healthcare & Pharmaceuticals",
    "ISPH":  "Healthcare & Pharmaceuticals",
    # Food & Beverages
    "JUFO":  "Food & Beverages",
    # Metals & Mining
    "EGAL":  "Metals & Mining",
    # Telecommunications
    "ETEL":  "Telecommunications",
    # Textiles
    "ORWE":  "Textiles",
    # Tobacco & Consumer Goods
    "EAST":  "Tobacco & Consumer Goods",
    # Diversified Holding
    "GBCO":  "Diversified Holding",
}

# Static fallbacks -- used only when an older trace has no `dynamic_bands` block.
STATIC_NEUTRAL_BAND_PCT = {"short": 1.0, "medium": 2.5, "large": 5.0}
STATIC_MAGNITUDE_BANDS_PCT = {
    "short":  {"small": [0.0, 1.0],  "medium": [1.0,  3.0],  "large": [3.0,  None]},
    "medium": {"small": [0.0, 2.5],  "medium": [2.5,  6.0],  "large": [6.0,  None]},
    "large":  {"small": [0.0, 5.0],  "medium": [5.0, 12.0],  "large": [12.0, None]},
}

# -----------------------------------------------------------------------------
#  Return computation  (with inclusive flag for trading-hours rule)
# -----------------------------------------------------------------------------

def cumulative_return_pct(
    returns: pd.DataFrame,
    ticker: str,
    event_date: pd.Timestamp,
    n_trading_days: int,
    inclusive: bool = False,
) -> tuple[float | None, pd.DatetimeIndex | None]:
    """
    Cumulative return of the company over the horizon window, in percent.

    inclusive=False (default)  -> window starts STRICTLY AFTER event_date
    inclusive=True             -> window STARTS AT event_date (news during hours)

    Returns (cum_return_pct, window_dates) so the caller can do dividend
    adjustments against the actual dates in the window.
    """
    if ticker not in returns.columns:
        return None, None

    if inclusive:
        future = returns.loc[returns.index >= event_date]
    else:
        future = returns.loc[returns.index > event_date]
    if future.empty:
        return None, None
    window = future.head(n_trading_days)
    if len(window) < n_trading_days:
        return None, None

    stock_window = window[ticker].dropna()
    if len(stock_window) < n_trading_days:
        return None, None

    cum = (np.exp(stock_window.sum()) - 1.0) * 100.0
    return float(cum), stock_window.index


# -----------------------------------------------------------------------------
#  Per-trace bands + neutral band + dividend adjustment
# -----------------------------------------------------------------------------

def _trace_bands(pred: dict) -> tuple[dict, dict]:
    """
    Return (neutral_band_pct, magnitude_bands_pct) sourced from the trace's
    dynamic_bands if present, falling back to static bands otherwise.
    """
    db = (pred.get("context") or {}).get("dynamic_bands") or {}
    return (
        db.get("neutral_band_pct")    or STATIC_NEUTRAL_BAND_PCT,
        db.get("magnitude_bands_pct") or STATIC_MAGNITUDE_BANDS_PCT,
    )


def _dividend_adjustment_pct(
    pred: dict,
    window_dates: pd.DatetimeIndex | None,
) -> float:
    """
    If news_type=dividend AND the trace carries dividend_info.yield_pct AND
    the ex-div date falls inside the horizon window, return the yield % to
    ADD BACK to the AR (neutralising the mechanical ex-div drop).
    Returns 0.0 otherwise.
    """
    primary = (pred.get("critic_agent") or {}).get("output", {}).get("primary_company") or {}
    if primary.get("news_type") != "dividend":
        return 0.0
    info = primary.get("dividend_info") or {}
    yield_pct = info.get("yield_pct")
    if yield_pct is None:
        return 0.0
    ex_div_date_str = info.get("ex_div_date")
    if not ex_div_date_str:
        # Conservative: if we know yield but not the date, do NOT adjust --
        # the article may not have stated the date and the ex-div day may
        # fall outside the window entirely.
        return 0.0
    try:
        ex_div_ts = pd.Timestamp(ex_div_date_str)
    except (ValueError, TypeError):
        return 0.0
    if window_dates is None or len(window_dates) == 0:
        return 0.0
    if ex_div_ts < window_dates[0] or ex_div_ts > window_dates[-1]:
        return 0.0
    return float(yield_pct)


# -----------------------------------------------------------------------------
#  Classification helpers (per-horizon, per-trace bands)
# -----------------------------------------------------------------------------

def directional_hit(
    predicted_direction: str, actual_ar_pct: float, horizon: str, neutral_band: dict
) -> bool | None:
    band = neutral_band.get(horizon, STATIC_NEUTRAL_BAND_PCT[horizon])
    if predicted_direction == "neutral":  return abs(actual_ar_pct) <= band
    if predicted_direction == "up":       return actual_ar_pct >  band
    if predicted_direction == "down":     return actual_ar_pct < -band
    return None


def magnitude_in_band(
    predicted_magnitude: str, actual_ar_pct: float, horizon: str, mag_bands: dict
) -> bool | None:
    bands = (mag_bands.get(horizon) or {})
    if predicted_magnitude not in bands:
        return None
    lo, hi = bands[predicted_magnitude]
    abs_ar = abs(actual_ar_pct)
    hi_eff = float("inf") if hi is None else hi
    return lo <= abs_ar < hi_eff


def observed_magnitude(actual_ar_pct: float, horizon: str, mag_bands: dict) -> str:
    abs_ar = abs(actual_ar_pct)
    bands  = mag_bands.get(horizon) or STATIC_MAGNITUDE_BANDS_PCT[horizon]
    for name, (lo, hi) in bands.items():
        hi_eff = float("inf") if hi is None else hi
        if lo <= abs_ar < hi_eff:
            return name
    return "large"


def observed_direction(actual_ar_pct: float, horizon: str, neutral_band: dict) -> str:
    band = neutral_band.get(horizon, STATIC_NEUTRAL_BAND_PCT[horizon])
    if abs(actual_ar_pct) <= band:
        return "neutral"
    return "up" if actual_ar_pct > 0 else "down"


# -----------------------------------------------------------------------------
#  Scoring a single prediction file  ->  one row per horizon
# -----------------------------------------------------------------------------

def score_prediction(
    pred: dict, returns: pd.DataFrame, sector_returns: pd.DataFrame | None = None
) -> list[dict]:
    primary        = pred["critic_agent"]["output"]["primary_company"]
    ticker         = primary["ticker"]
    news_type      = primary.get("news_type")
    already_priced = primary.get("already_priced_in")
    ta_signals     = primary.get("ta_signals_cited") or []
    per_horizon    = primary.get("per_horizon", {}) or {}

    event_date_str = pred["article_date"]
    event_date     = pd.Timestamp(event_date_str)

    timing    = (pred.get("context") or {}).get("news_timing") or {}
    inclusive = bool(timing.get("in_trading_hours"))

    sector     = SECTOR_MAP.get(ticker)
    neutral_band, mag_bands = _trace_bands(pred)

    rows: list[dict] = []
    for h in HORIZONS:
        h_pred = per_horizon.get(h) or {}
        direction  = h_pred.get("direction")
        magnitude  = h_pred.get("magnitude")
        confidence = h_pred.get("confidence")

        n_days = HORIZON_TRADING_DAYS[h]
        actual_ret, window_dates = cumulative_return_pct(
            returns, ticker, event_date, n_days, inclusive=inclusive
        )

        # Sector return over the same window (for abnormal-return column)
        sector_ret: float | None = None
        if sector_returns is not None and sector is not None and actual_ret is not None:
            sector_ret, _ = cumulative_return_pct(
                sector_returns, sector, event_date, n_days, inclusive=inclusive
            )

        abnormal_ret = (
            round(actual_ret - sector_ret, 3)
            if actual_ret is not None and sector_ret is not None
            else None
        )

        dividend_adj   = _dividend_adjustment_pct(pred, window_dates) if actual_ret is not None else 0.0
        actual_ret_adj = actual_ret + dividend_adj if actual_ret is not None else None

        base = {
            "ticker":               ticker,
            "sector":               sector,
            "event_date":           event_date_str,
            "horizon":              h,
            "news_type":            news_type,
            "in_trading_hours":     inclusive,
            "already_priced_in":    already_priced,
            "ta_signals_cited":     "|".join(ta_signals) if ta_signals else "",
            "predicted_direction":  direction,
            "predicted_magnitude":  magnitude,
            "confidence":           confidence,
            "actual_return_pct":    None if actual_ret is None else round(actual_ret, 3),
            "sector_return_pct":    None if sector_ret is None else round(sector_ret, 3),
            "abnormal_return_pct":  abnormal_ret,
            "dividend_adj_pct":     round(dividend_adj, 3),
            "actual_return_adj_pct": None if actual_ret_adj is None else round(actual_ret_adj, 3),
        }

        if actual_ret_adj is None or direction is None or magnitude is None:
            rows.append({
                **base,
                "observed_direction": None,
                "observed_magnitude": None,
                "directional_hit":    None,
                "magnitude_hit":      None,
                "status": "no_data" if actual_ret is None else "no_prediction",
            })
            continue

        dir_hit  = directional_hit(direction, actual_ret_adj, h, neutral_band)
        mag_band = magnitude_in_band(magnitude, actual_ret_adj, h, mag_bands)
        mag_hit  = bool(dir_hit) and bool(mag_band) if mag_band is not None else None

        rows.append({
            **base,
            "observed_direction": observed_direction(actual_ret_adj, h, neutral_band),
            "observed_magnitude": observed_magnitude(actual_ret_adj, h, mag_bands),
            "directional_hit":    bool(dir_hit),
            "magnitude_hit":      mag_hit,
            "status":             "scored",
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
    if c is None:  return None
    if c < 0.4:    return "0.0-0.4"
    if c < 0.6:    return "0.4-0.6"
    if c < 0.8:    return "0.6-0.8"
    return "0.8-1.0"


def _confidence_aggregation(records: list[dict]) -> dict:
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
    buckets: dict = defaultdict(lambda: {"n": 0, "dir_hits": 0, "mag_hits": 0})
    for r in records:
        if r["status"] != "scored":
            continue
        for sig in (r.get("ta_signals_cited") or "").split("|"):
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
        "by_in_trading_hours":          _hit_rate(scored, "in_trading_hours"),
        "horizon_x_news_type":          _hit_rate_by_two(scored, "horizon", "news_type"),
        "by_confidence_per_horizon":    _confidence_aggregation(scored),
        "by_ta_signal":                 _ta_signal_hit_rate(scored),
        "by_sector":                    _hit_rate(scored, "sector"),
        "horizon_x_sector":             _hit_rate_by_two(scored, "horizon", "sector"),
        "by_ticker":                    _hit_rate(scored, "ticker"),
        "horizon_x_ticker":             _hit_rate_by_two(scored, "horizon", "ticker"),
    }


# -----------------------------------------------------------------------------
#  CLI
# -----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions-dir", type=Path, default=Path("outputs/predictions_v5"))
    p.add_argument("--returns",         type=Path, default=Path("returns_daily.parquet"))
    p.add_argument("--sector-returns",  type=Path, default=Path("sector_returns_daily.parquet"))
    p.add_argument("--out",             type=Path, default=Path("outputs/evaluation_v5"))
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

    df = pd.DataFrame(records)
    df.to_csv(args.out / "scorecard.csv", index=False)

    agg = aggregate(records)
    with open(args.out / "aggregate.json", "w", encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False, indent=2, default=str)

    print(f"\nScored {agg['n_scored']}/{agg['n_total_rows']} (article x horizon) rows "
          f"from {n_files} prediction file(s).")
    print(f"  no_data       : {agg['n_no_data']}    (missing future prices)")
    print(f"  no_prediction : {agg['n_no_prediction']}")
    print(f"\nOverall directional hit rate : {agg['overall_directional_hit_rate']}")
    print(f"Overall magnitude   hit rate : {agg['overall_magnitude_hit_rate']}")

    print("\nBy horizon:")
    for h in HORIZONS:
        row = agg["by_horizon"].get(h, {})
        print(f"  {h:<7}  n={row.get('n', 0):<4}  "
              f"dir={row.get('directional_hit_rate')}  "
              f"mag={row.get('magnitude_hit_rate')}")

    if agg["by_ticker"]:
        print("\nBy company (ticker):")
        for ticker, m in sorted(agg["by_ticker"].items(), key=lambda x: x[0]):
            sec = SECTOR_MAP.get(ticker, "Unknown")
            print(f"  {ticker:<8}  [{sec:<38}]  n={m['n']:<4}  "
                  f"dir={m['directional_hit_rate']}  mag={m['magnitude_hit_rate']}")

    if agg["by_sector"]:
        print("\nBy sector:")
        for sec, m in sorted(agg["by_sector"].items(), key=lambda x: x[0]):
            print(f"  {sec:<38}  n={m['n']:<4}  "
                  f"dir={m['directional_hit_rate']}  mag={m['magnitude_hit_rate']}")

    if agg["by_ta_signal"]:
        print("\nTop TA signals by directional hit rate (n>=3):")
        ranked = sorted(
            ((sig, m) for sig, m in agg["by_ta_signal"].items() if m["n"] >= 3),
            key=lambda x: (x[1]["directional_hit_rate"] or 0),
            reverse=True,
        )
        for sig, m in ranked[:10]:
            print(f"  {sig:<28}  n={m['n']:<3}  "
                  f"dir={m['directional_hit_rate']}  mag={m['magnitude_hit_rate']}")

    print(f"\nWrote: {args.out / 'scorecard.csv'}")
    print(f"Wrote: {args.out / 'aggregate.json'}")


if __name__ == "__main__":
    main()
