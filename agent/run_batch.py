"""
Batch runner for the EGX News-Impact Agent v2.

Processes every article in all_news_by_date_mubasher.json through the full
LLM pipeline, saves one prediction JSON per article, then produces a combined
evaluation report covering:

  • Direction accuracy   — predicted up/down/neutral vs realised returns
                           (uses returns_daily.parquet + sector_returns_daily.parquet
                            when available; falls back gracefully when absent)
  • Absolute return eval — realised % return per ticker/horizon from the raw
                            prices JSON (no parquet dependency)
  • Latency statistics   — mean, median, p95, p99 wall-clock time per article

Usage
-----
  python -m agent.run_batch                          # all 1 238 articles, concurrency=3
  python -m agent.run_batch --limit 20               # quick smoke-test
  python -m agent.run_batch --resume                 # skip already-saved predictions
  python -m agent.run_batch --concurrency 5          # higher throughput

Outputs (written under agent/outputs/ by default)
--------------------------------------------------
  predictions/          one JSON per article (evaluate.py-compatible)
  evaluation/
    aggregate.json      combined evaluation (direction + returns + latency)
    scorecard.csv       per-row direction scores (requires parquet files)
    returns_detail.csv  per-row absolute-return scores (always produced)
    latency.csv         per-article latency log
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

# ── bootstrap sys.path so agent.* and root-level modules resolve ──────────────
import sys
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent import pipeline                           # noqa: E402
from agent.context import build_context              # noqa: E402
from agent.evaluate import SECTOR_MAP, aggregate, score_prediction  # noqa: E402
from agent.config import ACTIVE_HORIZONS, HORIZON_DAYS              # noqa: E402

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_NEWS    = Path("agent/news/all_news_by_date_mubasher.json")
DEFAULT_PRICES  = Path("agent/stock_prices/egyptian_stocks_2020_2025.json")
DEFAULT_OUT     = Path("agent/outputs")
DEFAULT_CONCURRENCY = 3

# ── change this to cap how many articles are processed ────────────────────────
# Set to None to run ALL articles (full batch).
# Set to any positive integer for a partial run, e.g. 5, 20, 100.
DEFAULT_LIMIT: int | None = 20


# ─────────────────────────────────────────────────────────────────────────────
# Article normalisation
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(raw: dict) -> dict:
    """Map a Mubasher article record to the format pipeline.run() expects."""
    return {
        "ticker":       raw["symbol"],
        "title":        raw.get("title", ""),
        "body":         raw.get("body", ""),
        "published_at": raw.get("date", ""),
        "datetime":     raw.get("datetime", ""),
        "time":         raw.get("time", ""),
        "source":       raw.get("source", "مباشر"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Single-article processor
# ─────────────────────────────────────────────────────────────────────────────

async def _process_one(
    raw: dict,
    sem: asyncio.Semaphore,
    pred_dir: Path,
    resume: bool,
) -> dict:
    """
    Run one article through the pipeline.

    Returns a metadata dict:
      status   : "ok" | "error" | "skipped"
      latency_s: wall-clock seconds (None when skipped)
      error    : error message string (None on success)
    """
    ticker     = raw["symbol"]
    event_date = raw.get("date", "")[:10]
    art_id     = str(raw.get("id", ""))
    fname      = f"{ticker}_{event_date.replace('-', '')}_{art_id}.json"
    out_path   = pred_dir / fname

    meta: dict[str, Any] = {
        "ticker":     ticker,
        "event_date": event_date,
        "article_id": art_id,
        "file":       fname,
        "status":     None,
        "latency_s":  None,
        "error":      None,
    }

    if resume and out_path.exists():
        meta["status"] = "skipped"
        return meta

    article = _normalize(raw)

    async with sem:
        t0 = time.perf_counter()
        try:
            response = await pipeline.run(article)
            latency_s = time.perf_counter() - t0
        except Exception as exc:
            meta["status"]    = "error"
            meta["latency_s"] = round(time.perf_counter() - t0, 3)
            meta["error"]     = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
            return meta

    # build_context is fast (deterministic, uses cached JSON) — call it to
    # extract ATR values and the in_trading_hours flag for the prediction file.
    try:
        ctx        = build_context(ticker, event_date, article)
        ticker_atr = ctx["ticker_context"].get("atr_14d")
        sector_atr = ctx["sector_context"].get("atr_14d")
        in_hours   = ctx["news_timing"].get("in_trading_hours", False)
    except Exception:
        ticker_atr = sector_atr = None
        in_hours   = False

    # ── prediction record — evaluate.py-compatible format ────────────────────
    pred: dict[str, Any] = {
        "ticker":           ticker,
        "event_date":       event_date,
        "in_trading_hours": in_hours,
        "event_type":       response.event_type,
        "sentiment_label":  response.sentiment_label or "neutral",
        "critic_invoked":   response.critic_invoked,
        "context": {
            "ticker_atr": ticker_atr,
            "sector_atr": sector_atr,
        },
        "outlook": {
            h: {
                "horizon_days":     hor.horizon_days,
                "ticker":           hor.ticker,
                "sector":           hor.sector,
                "implied_abnormal": hor.implied_abnormal,
                "explanation":      hor.explanation,   # plain-Arabic per-horizon outlook
            }
            for h, hor in response.outlook.items()
        },
        # full user-facing Arabic explanation (headline + narrative + per-horizon)
        "explanation": response.explanation.model_dump(),
        # extra fields for traceability (ignored by evaluate.py)
        "latency_s":    round(latency_s, 3),
        "processed_at": response.processed_at,
        "article": {
            "id":     art_id,
            "title":  article["title"],
            "source": article.get("source", ""),
        },
    }

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(pred, fh, ensure_ascii=False, indent=2, default=str)

    meta["status"]    = "ok"
    meta["latency_s"] = round(latency_s, 3)
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# Latency evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _latency_stats(metas: list[dict]) -> dict:
    lats = sorted(m["latency_s"] for m in metas if m["status"] == "ok" and m["latency_s"])
    if not lats:
        return {}
    n = len(lats)
    return {
        "n":        n,
        "mean_s":   round(statistics.mean(lats), 3),
        "median_s": round(statistics.median(lats), 3),
        "min_s":    round(lats[0], 3),
        "max_s":    round(lats[-1], 3),
        "p95_s":    round(lats[int(0.95 * n)], 3),
        "p99_s":    round(lats[min(int(0.99 * n), n - 1)], 3),
        "total_wall_s": round(sum(lats), 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Absolute-return evaluation (prices JSON)
# ─────────────────────────────────────────────────────────────────────────────

def _build_price_index(prices_json: dict) -> dict[str, pd.Series]:
    """Return {ticker: pd.Series of Close prices indexed by trading date}."""
    idx: dict[str, pd.Series] = {}
    for ticker, data in prices_json.items():
        rows = data.get("prices", [])
        if not rows:
            continue
        df = pd.DataFrame(rows)[["Date", "Close"]].copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        idx[ticker] = df["Close"].astype(float)
    return idx


def _realised_ret_detail(
    series: pd.Series,
    event_date_str: str,
    n_days: int,
    in_hours: bool,
) -> dict | None:
    """
    Compute return for an N-trading-day window and return all price details.

    Anchoring (consistent with the parquet log-return evaluator):
      start_day = first session reacting to the news
                  • in_hours=True  → the event date itself (t+0)
                  • in_hours=False → the next trading session
      Return = (close_at_end_of_window / close_BEFORE_window - 1) * 100

    Returns a dict with keys:
      anchor_date, anchor_price  — last close BEFORE the window (pre-news)
      start_date,  start_price   — first session of reaction window
      end_date,    end_price     — last session of reaction window
      ret_pct                    — cumulative % return (anchor→end)
    Returns None when the window falls outside available data.
    """
    event = pd.Timestamp(event_date_str)
    idx = series.index
    side = "left" if in_hours else "right"
    start_pos  = int(idx.searchsorted(event, side=side))
    anchor_pos = start_pos - 1
    end_pos    = start_pos + n_days - 1
    if anchor_pos < 0 or end_pos >= len(idx):
        return None
    p_anchor = float(series.iloc[anchor_pos])
    p_start  = float(series.iloc[start_pos])
    p_end    = float(series.iloc[end_pos])
    if p_anchor <= 0:
        return None
    return {
        "anchor_date":  idx[anchor_pos].strftime("%Y-%m-%d"),
        "anchor_price": round(p_anchor, 4),
        "start_date":   idx[start_pos].strftime("%Y-%m-%d"),
        "start_price":  round(p_start, 4),
        "end_date":     idx[end_pos].strftime("%Y-%m-%d"),
        "end_price":    round(p_end, 4),
        "ret_pct":      round((p_end / p_anchor - 1.0) * 100.0, 4),
    }


def _obs_direction(ret: float, band: float) -> str:
    if abs(ret) <= band:
        return "neutral"
    return "up" if ret > 0 else "down"


def _atr_band(ticker_atr: float | None, horizon: str) -> float:
    n = HORIZON_DAYS[horizon]
    if ticker_atr and ticker_atr > 0:
        return round(0.5 * ticker_atr * math.sqrt(n), 2)
    return {"short": 1.0, "medium": 1.5, "large": 4.0}[horizon]


def eval_absolute_returns(pred_dir: Path, prices_json: dict) -> tuple[dict, pd.DataFrame]:
    """
    Score every saved prediction against realised prices.

    Returns (summary_dict, detail_dataframe).
    """
    price_idx = _build_price_index(prices_json)
    rows: list[dict] = []

    for path in sorted(pred_dir.glob("*.json")):
        try:
            with open(path, encoding="utf-8") as fh:
                pred = json.load(fh)
        except Exception:
            continue

        ticker     = pred.get("ticker", "")
        event_date = pred.get("event_date", "")
        in_hours   = pred.get("in_trading_hours", False)
        ticker_atr = (pred.get("context") or {}).get("ticker_atr")

        series = price_idx.get(ticker)

        for h in ACTIVE_HORIZONS:
            n_days  = HORIZON_DAYS[h]
            h_pred  = (pred.get("outlook") or {}).get(h) or {}
            t_pred  = h_pred.get("ticker") or {}
            pred_dir_val = t_pred.get("direction")
            pred_conf    = t_pred.get("confidence")
            pred_mag     = t_pred.get("magnitude")

            detail = _realised_ret_detail(series, event_date, n_days, in_hours) if series is not None else None
            band   = _atr_band(ticker_atr, h)

            actual_ret = detail["ret_pct"] if detail else None

            row: dict[str, Any] = {
                "ticker":        ticker,
                "sector":        SECTOR_MAP.get(ticker),
                "event_date":    event_date,
                "horizon":       h,
                "horizon_days":  n_days,
                "event_type":    pred.get("event_type"),
                "sentiment":     pred.get("sentiment_label"),
                "pred_dir":      pred_dir_val,
                "pred_mag":      pred_mag,
                "pred_conf":     pred_conf,
                # price window detail — key for debugging
                "anchor_date":   detail["anchor_date"]  if detail else None,
                "anchor_price":  detail["anchor_price"] if detail else None,
                "start_date":    detail["start_date"]   if detail else None,
                "start_price":   detail["start_price"]  if detail else None,
                "end_date":      detail["end_date"]     if detail else None,
                "end_price":     detail["end_price"]    if detail else None,
                "actual_ret_pct": actual_ret,
                "neutral_band":  band,
                "obs_dir":       None,
                "hit":           None,
                "file":          path.name,
            }

            if actual_ret is not None and pred_dir_val:
                obs = _obs_direction(actual_ret, band)
                row["obs_dir"] = obs
                row["hit"]     = (pred_dir_val == obs)

            rows.append(row)

    detail_df = pd.DataFrame(rows) if rows else pd.DataFrame()

    if detail_df.empty:
        return {"error": "no predictions found"}, detail_df

    scored = detail_df[detail_df["hit"].notna()].copy()

    summary: dict[str, Any] = {
        "n_predictions": len(detail_df),
        "n_scored":      len(scored),
        "n_no_data":     int(detail_df["actual_ret_pct"].isna().sum()),
    }

    if scored.empty:
        summary["error"] = "no scored rows (all predictions lack future price data)"
        return summary, detail_df

    # ── overall hit rate ──────────────────────────────────────────────────────
    summary["overall"] = {
        "hit_rate": round(float(scored["hit"].mean()), 3),
        "mean_abs_ret_pct": round(float(scored["actual_ret_pct"].abs().mean()), 3),
    }

    # ── by horizon (the primary breakdown) ────────────────────────────────────
    by_h: dict = {}
    for h in ACTIVE_HORIZONS:
        hdf = scored[scored["horizon"] == h]
        if hdf.empty:
            continue
        n_hit = int(hdf["hit"].sum())
        n_miss = len(hdf) - n_hit

        # distribution of predicted directions for this horizon
        pred_dist = {d: int((hdf["pred_dir"] == d).sum()) for d in ("up", "down", "neutral")}
        obs_dist  = {d: int((hdf["obs_dir"] == d).sum())  for d in ("up", "down", "neutral")}

        # hit-rate split by predicted direction (precision of each call)
        dir_hits: dict = {}
        for d in ("up", "down", "neutral"):
            grp = hdf[hdf["pred_dir"] == d]
            if not grp.empty:
                dir_hits[d] = {
                    "n":        len(grp),
                    "hit_rate": round(float(grp["hit"].mean()), 3),
                    "mean_ret_pct": round(float(grp["actual_ret_pct"].mean()), 3),
                }

        # always-up baseline on this horizon, for comparison
        bl_up = round(float((hdf["obs_dir"] == "up").mean()), 3)

        by_h[h] = {
            "horizon_days":     HORIZON_DAYS[h],
            "n":                len(hdf),
            "hit_rate":         round(float(hdf["hit"].mean()), 3),
            "baseline_always_up": bl_up,
            "mean_abs_ret_pct": round(float(hdf["actual_ret_pct"].abs().mean()), 3),
            "mean_ret_correct": round(float(hdf[hdf["hit"]]["actual_ret_pct"].mean()), 3) if n_hit else None,
            "mean_ret_wrong":   round(float(hdf[~hdf["hit"]]["actual_ret_pct"].mean()), 3) if n_miss else None,
            "ret_p25":          round(float(hdf["actual_ret_pct"].quantile(0.25)), 3),
            "ret_p50":          round(float(hdf["actual_ret_pct"].quantile(0.50)), 3),
            "ret_p75":          round(float(hdf["actual_ret_pct"].quantile(0.75)), 3),
            "ret_std":          round(float(hdf["actual_ret_pct"].std()), 3),
            "pred_distribution": pred_dist,
            "obs_distribution":  obs_dist,
            "hit_rate_by_predicted_direction": dir_hits,
        }
    summary["by_horizon"] = by_h

    # ── by predicted direction ────────────────────────────────────────────────
    by_dir: dict = {}
    for d in ("up", "down", "neutral"):
        grp = scored[scored["pred_dir"] == d]
        if grp.empty:
            continue
        by_dir[d] = {
            "n":            len(grp),
            "hit_rate":     round(float(grp["hit"].mean()), 3),
            "mean_ret_pct": round(float(grp["actual_ret_pct"].mean()), 3),
        }
    summary["by_predicted_direction"] = by_dir

    # ── by sector ────────────────────────────────────────────────────────────
    by_sec: dict = {}
    for sec in sorted(scored["sector"].dropna().unique()):
        grp = scored[scored["sector"] == sec]
        by_sec[sec] = {
            "n":            len(grp),
            "hit_rate":     round(float(grp["hit"].mean()), 3),
            "mean_ret_pct": round(float(grp["actual_ret_pct"].mean()), 3),
        }
    summary["by_sector"] = by_sec

    return summary, detail_df


# ─────────────────────────────────────────────────────────────────────────────
# Direction evaluation (parquet-based)
# ─────────────────────────────────────────────────────────────────────────────

def eval_direction(pred_dir: Path) -> tuple[dict, list[dict]]:
    returns_path = _ROOT / "returns_daily.parquet"
    sector_path  = _ROOT / "sector_returns_daily.parquet"

    if not returns_path.exists() or not sector_path.exists():
        return {"error": "parquet files not found — run prepare_data.py first"}, []

    t_ret = pd.read_parquet(returns_path)
    t_ret.index = pd.to_datetime(t_ret.index)
    s_ret = pd.read_parquet(sector_path)
    s_ret.index = pd.to_datetime(s_ret.index)

    all_rows: list[dict] = []
    for path in sorted(pred_dir.glob("*.json")):
        try:
            with open(path, encoding="utf-8") as fh:
                pred = json.load(fh)
            rows = score_prediction(pred, t_ret, s_ret)
            for r in rows:
                r["file"] = path.name
            all_rows.extend(rows)
        except Exception as exc:
            print(f"  [WARN] {path.name}: {exc}")

    return aggregate(all_rows), all_rows


# ─────────────────────────────────────────────────────────────────────────────
# Main async orchestrator
# ─────────────────────────────────────────────────────────────────────────────

async def run_batch(
    news_path:   Path,
    prices_path: Path,
    out_dir:     Path,
    concurrency: int,
    limit:       int | None,
    resume:      bool,
) -> None:
    pred_dir = out_dir / "predictions"
    eval_dir = out_dir / "evaluation"
    pred_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    # ── load news ─────────────────────────────────────────────────────────────
    print(f"Loading news from {news_path} …")
    with open(news_path, encoding="utf-8") as fh:
        news_data = json.load(fh)

    articles: list[dict] = []
    for _date, day_arts in sorted(news_data["by_date"].items()):
        articles.extend(day_arts)

    if limit:
        articles = articles[:limit]

    total = len(articles)
    print(f"Articles to process: {total}  (concurrency={concurrency}, resume={resume})")

    # ── run pipeline ──────────────────────────────────────────────────────────
    sem   = asyncio.Semaphore(concurrency)
    tasks = [_process_one(a, sem, pred_dir, resume) for a in articles]

    metas:   list[dict] = []
    n_ok = n_err = n_skip = 0

    for coro in asyncio.as_completed(tasks):
        m = await coro
        metas.append(m)
        if m["status"] == "ok":
            n_ok += 1
            lat = m["latency_s"]
            done = n_ok + n_err + n_skip
            print(f"  [{done:>4}/{total}] OK      {m['ticker']:<8} {m['event_date']}  {lat:.1f}s")
        elif m["status"] == "error":
            n_err += 1
            done = n_ok + n_err + n_skip
            print(f"  [{done:>4}/{total}] ERROR   {m['ticker']:<8} {m['event_date']}  {m['error'][:70]}")
        else:
            n_skip += 1

    print(f"\nPipeline done: {n_ok} ok | {n_skip} skipped | {n_err} errors  (total {total})\n")

    # ── save per-article latency log ──────────────────────────────────────────
    lat_df = pd.DataFrame(metas)
    lat_df.to_csv(eval_dir / "latency.csv", index=False)

    latency = _latency_stats(metas)
    _print_latency(latency)

    # ── direction evaluation (parquet) ────────────────────────────────────────
    print("\nDirection evaluation (parquet log-returns) …")
    direction_agg, scorecard_rows = eval_direction(pred_dir)
    if scorecard_rows:
        pd.DataFrame(scorecard_rows).to_csv(eval_dir / "scorecard.csv", index=False)
        _print_direction(direction_agg)
    else:
        print(f"  {direction_agg.get('error', 'no rows')}")

    # ── absolute-return evaluation (prices JSON) ──────────────────────────────
    print("\nAbsolute-return evaluation (prices JSON) …")
    with open(prices_path, encoding="utf-8") as fh:
        prices_json = json.load(fh)
    abs_ret_agg, ret_detail_df = eval_absolute_returns(pred_dir, prices_json)
    if not ret_detail_df.empty:
        ret_detail_df.to_csv(eval_dir / "returns_detail.csv", index=False)
    _print_abs_returns(abs_ret_agg)
    _print_detail_table(ret_detail_df)

    # ── combined aggregate.json ───────────────────────────────────────────────
    combined = {
        "generated_at":          datetime.now(timezone.utc).isoformat(),
        "run": {
            "n_total":     total,
            "n_ok":        n_ok,
            "n_skipped":   n_skip,
            "n_errors":    n_err,
            "concurrency": concurrency,
            "news_path":   str(news_path),
            "prices_path": str(prices_path),
        },
        "latency":            latency,
        "direction_accuracy": direction_agg,
        "absolute_return":    abs_ret_agg,
    }
    agg_path = eval_dir / "aggregate.json"
    with open(agg_path, "w", encoding="utf-8") as fh:
        json.dump(combined, fh, ensure_ascii=False, indent=2, default=str)

    print(f"\nOutputs")
    print(f"  predictions     : {pred_dir}  ({n_ok + n_skip} files)")
    print(f"  aggregate.json  : {agg_path}")
    if (eval_dir / "scorecard.csv").exists():
        print(f"  scorecard.csv   : {eval_dir / 'scorecard.csv'}")
    if (eval_dir / "returns_detail.csv").exists():
        print(f"  returns_detail  : {eval_dir / 'returns_detail.csv'}")
    print(f"  latency.csv     : {eval_dir / 'latency.csv'}")


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_latency(s: dict) -> None:
    if not s:
        return
    print("\nLatency")
    print(f"  mean   : {s['mean_s']} s")
    print(f"  median : {s['median_s']} s")
    print(f"  p95    : {s['p95_s']} s")
    print(f"  p99    : {s['p99_s']} s")
    print(f"  min    : {s['min_s']} s   max: {s['max_s']} s")
    print(f"  total  : {s['total_wall_s']} s  ({s['n']} articles)")


def _print_direction(agg: dict) -> None:
    if "error" in agg:
        print(f"  {agg['error']}")
        return
    print(f"\nDirection accuracy (parquet)  —  {agg['n_scored']}/{agg['n_total']} scored")
    for target in ("ticker", "sector", "implied_abnormal"):
        t = agg.get(target, {})
        ov = t.get("overall", {})
        hr = ov.get("hit_rate")
        n  = ov.get("n")
        if hr is None:
            continue
        print(f"  {target:<22}: {hr:.3f}  (n={n})")
        for bl in ("baseline_always_up", "baseline_market", "baseline_sentiment"):
            bv = t.get(bl, {}).get("hit_rate")
            if bv is not None:
                print(f"    {bl:<28}: {bv:.3f}")
        if "by_horizon" in t:
            for h, m in t["by_horizon"].items():
                print(f"    {h:<8}: {m['hit_rate']:.3f}  (n={m['n']})")


def _print_abs_returns(s: dict) -> None:
    if "error" in s:
        print(f"  {s['error']}")
        return
    print(f"\nAbsolute return  —  {s.get('n_scored')}/{s.get('n_predictions')} scored  "
          f"({s.get('n_no_data')} no-data)")
    ov = s.get("overall", {})
    if ov:
        print(f"  overall hit_rate     : {ov['hit_rate']:.3f}")
        print(f"  mean |return|        : {ov['mean_abs_ret_pct']:.2f}%")
    print("  per-horizon:")
    for h, m in (s.get("by_horizon") or {}).items():
        lift = m["hit_rate"] - m.get("baseline_always_up", 0.0)
        print(f"    {h:<7} ({m['horizon_days']:>2}d)  n={m['n']:<4}  "
              f"hit={m['hit_rate']:.3f}  base_up={m.get('baseline_always_up', 0):.3f}  "
              f"lift={lift:+.3f}  mean|ret|={m['mean_abs_ret_pct']:.2f}%  "
              f"p25={m['ret_p25']:+.2f}%  p50={m['ret_p50']:+.2f}%  p75={m['ret_p75']:+.2f}%  "
              f"std={m['ret_std']:.2f}%")
        if m.get("mean_ret_correct") is not None:
            print(f"              correct preds mean_ret={m['mean_ret_correct']:+.2f}%  "
                  f"wrong preds mean_ret={m.get('mean_ret_wrong', 0):+.2f}%")
        pd_ = m.get("pred_distribution", {})
        obs_ = m.get("obs_distribution", {})
        print(f"            predicted : up={pd_.get('up',0):>3}  down={pd_.get('down',0):>3}  "
              f"neutral={pd_.get('neutral',0):>3}")
        print(f"            observed  : up={obs_.get('up',0):>3}  down={obs_.get('down',0):>3}  "
              f"neutral={obs_.get('neutral',0):>3}")
        for d, dm in (m.get("hit_rate_by_predicted_direction") or {}).items():
            print(f"              when predicted {d:<7}: hit={dm['hit_rate']:.3f}  "
                  f"n={dm['n']:>3}  mean_ret={dm['mean_ret_pct']:+.2f}%")
    by_dir = s.get("by_predicted_direction") or {}
    if by_dir:
        print("  overall by predicted direction:")
        for d, m in by_dir.items():
            print(f"    {d:<7}: hit={m['hit_rate']:.3f}  n={m['n']:>3}  mean_ret={m['mean_ret_pct']:+.2f}%")
    by_sec = s.get("by_sector") or {}
    if by_sec:
        print("  by sector:")
        for sec, m in by_sec.items():
            print(f"    {sec:<30}: hit={m['hit_rate']:.3f}  n={m['n']:>3}  mean_ret={m['mean_ret_pct']:+.2f}%")


def _print_detail_table(df: pd.DataFrame) -> None:
    """Print a per-article, per-horizon price detail table for easy debugging."""
    if df.empty:
        return
    print("\n" + "─" * 120)
    print("Per-article price detail (sorted by ticker → event_date → horizon)")
    print("─" * 120)
    hdr = (
        f"{'TICKER':<8} {'EVENT':>10}  {'HOR':>6}  "
        f"{'PRED':>7} {'OBS':>7} {'RET%':>7}  "
        f"{'BAND%':>5}  {'HIT':>3}  "
        f"{'ANCHOR_DATE':>11} {'P_ANCHOR':>9}  "
        f"{'START_DATE':>10} {'P_START':>8}  "
        f"{'END_DATE':>10} {'P_END':>8}"
    )
    print(hdr)
    print("─" * 120)
    sort_cols = [c for c in ("ticker", "event_date", "horizon") if c in df.columns]
    rows = df.sort_values(sort_cols) if sort_cols else df
    for _, r in rows.iterrows():
        hit_mark = "✓" if r.get("hit") is True else ("✗" if r.get("hit") is False else "─")
        ret_str  = f"{r['actual_ret_pct']:+.2f}%" if pd.notna(r.get("actual_ret_pct")) else "  N/A  "
        band_str = f"{r['neutral_band']:.2f}" if pd.notna(r.get("neutral_band")) else "  ─  "
        p_anchor = f"{r['anchor_price']:.3f}" if pd.notna(r.get("anchor_price")) else "    ─    "
        p_start  = f"{r['start_price']:.3f}"  if pd.notna(r.get("start_price"))  else "    ─   "
        p_end    = f"{r['end_price']:.3f}"    if pd.notna(r.get("end_price"))    else "    ─   "
        anch_dt  = str(r.get("anchor_date", "─"))[:10]
        st_dt    = str(r.get("start_date",  "─"))[:10]
        en_dt    = str(r.get("end_date",    "─"))[:10]
        print(
            f"{str(r.get('ticker','')):<8} {str(r.get('event_date',''))[:10]:>10}  "
            f"{str(r.get('horizon',''))[:6]:>6}  "
            f"{str(r.get('pred_dir','─'))[:7]:>7} {str(r.get('obs_dir','─'))[:7]:>7} {ret_str:>7}  "
            f"{band_str:>5}  {hit_mark:>3}  "
            f"{anch_dt:>11} {p_anchor:>9}  "
            f"{st_dt:>10} {p_start:>8}  "
            f"{en_dt:>10} {p_end:>8}"
        )
    print("─" * 120)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Batch runner for EGX News-Impact Agent v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--news",        type=Path, default=DEFAULT_NEWS,
                   help="Path to all_news_by_date_mubasher.json")
    p.add_argument("--prices",      type=Path, default=DEFAULT_PRICES,
                   help="Path to egyptian_stocks_2020_2025.json")
    p.add_argument("--out",         type=Path, default=DEFAULT_OUT,
                   help="Output root directory (default: agent/outputs)")
    p.add_argument("--concurrency", type=int,  default=DEFAULT_CONCURRENCY,
                   help="Max concurrent pipeline calls (default: 3)")
    p.add_argument("--limit",       type=int,  default=DEFAULT_LIMIT,
                   help="Cap on articles to process (default: DEFAULT_LIMIT at top of file)")
    p.add_argument("--resume",      action="store_true",
                   help="Skip articles whose prediction file already exists")
    args = p.parse_args()

    asyncio.run(run_batch(
        news_path=args.news,
        prices_path=args.prices,
        out_dir=args.out,
        concurrency=args.concurrency,
        limit=args.limit,
        resume=args.resume,
    ))


if __name__ == "__main__":
    main()
