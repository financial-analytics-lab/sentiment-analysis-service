# pipeline3_vote.py
"""
EXPERIMENTAL -- multi-Claude voting layer on top of the main pipeline (v2.1).

This script is OPTIONAL and runs INDEPENDENTLY of pipeline3.py. It:
  1. Reads a finished prediction trace from outputs/predictions_v2/
  2. Spawns N parallel Claude voters via concurrent.futures (NOT LangChain --
     LangChain/LangGraph add overhead for fan-out/fan-in like this; raw threads
     are faster). Each voter sees the same evidence + the critic's prediction
     and casts a BUY / HOLD / SELL + entry-price-range vote.
  3. Aggregates: majority on action, median on entry price, mean on confidence.
  4. Writes a voting summary to outputs/voting_v2/{trace_name}.json.

Latency profile (per trace):
    serial   : N x ~Claude_latency        ~30-60s typical
    parallel : 1 x ~Claude_latency        ~6-12s typical
    proxy 503: fails fast (no retry storm)

How to use:
    # Run after pipeline3.py has produced traces:
    python pipeline3_vote.py                       # vote on every trace in v2
    python pipeline3_vote.py EMFD_2025-01-02_b5fd0aa7.json   # vote on one file

Notes:
  - Uses temperature=0.5 (deterministic at 0 collapses all 5 votes to one
    answer; same-model voting needs >0 to produce diversity).
  - For REAL ensemble diversity, swap CLAUDE_MODEL_LIST to a multi-MODEL
    list spanning vendors (Anthropic + Groq Llama + Groq Qwen). The code
    is structured so swapping the per-voter API call is a one-function edit.
  - Failures are tolerated: if K of N voters fail, the result still
    aggregates over the remaining (N-K) successful votes.
"""

# =============================================================================
#  TOP-LEVEL CONTROLS
# =============================================================================

INPUT_DIR        = "outputs/predictions_v2"
OUTPUT_DIR       = "outputs/voting_v2"

N_VOTERS         = 5          # how many parallel votes per trace
VOTER_TEMP       = 0.5        # must be > 0 for diversity (same model)
VOTER_MAX_TOKENS = 1200       # voter output is small (action + range + reason)
VOTER_TIMEOUT_S  = 120        # per-voter HTTP timeout

# =============================================================================

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import median
from typing import Any

import requests

from config import CLAUDE_API_KEY, CLAUDE_BASE_URL, CLAUDE_MODEL


# Reuse the lenient JSON parser & think-tag stripping from pipeline3 if present;
# otherwise inline a minimal version so this file can run standalone.
try:
    from pipeline3 import _parse_json as _pipeline_parse_json   # type: ignore
    _parse_json = _pipeline_parse_json
except Exception:
    _THINK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)
    try:
        from json_repair import repair_json as _repair_json  # type: ignore
        _HAS_JR = True
    except ImportError:
        _HAS_JR = False

    def _parse_json(raw: str) -> dict:
        raw = _THINK_RE.sub("", raw.strip()).strip()
        for f in ("```json", "```"):
            if raw.startswith(f):
                raw = raw[len(f):]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()
        s, e = raw.find("{"), raw.rfind("}")
        if s != -1 and e != -1:
            raw = raw[s:e + 1]
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            if _HAS_JR:
                try:
                    return json.loads(_repair_json(raw, return_objects=False))
                except Exception:
                    pass
            raise


# =============================================================================
#  Voter prompt
# =============================================================================

def _build_voter_prompt(trace: dict) -> str:
    """
    Compact prompt: just enough context for a buy/hold/sell + entry-price call.
    Reuses what the critic already concluded; the voter's job is to AGREE,
    DISAGREE, or REFINE the entry price.
    """
    primary = trace["critic_agent"]["output"]["primary_company"]
    ctx     = trace.get("context") or {}
    ps      = ctx.get("price_state") or {}
    ts      = ctx.get("technical_signals") or {}
    dist    = (ts.get("distance_to_extremes") or {})
    db      = ctx.get("dynamic_bands") or {}

    last_close = ps.get("last_close")
    atr_pct    = ts.get("atr_pct_14d")
    h_short    = primary.get("per_horizon", {}).get("short",  {}) or {}
    h_medium   = primary.get("per_horizon", {}).get("medium", {}) or {}
    h_large    = primary.get("per_horizon", {}).get("large",  {}) or {}

    return f"""\
You are a discretionary equity trader voting on a single EGX30 stock decision.

=== TICKER & PRICE ===
Ticker      : {trace.get('ticker')}
Last close  : {last_close} EGP
ATR (14d)   : {atr_pct}%   (typical daily move for this ticker)
Distance to extremes:
  20d high : {dist.get('20d_high_distance_pct')}%   20d low: {dist.get('20d_low_distance_pct')}%
  52w high : {dist.get('252d_high_distance_pct')}%  52w low: {dist.get('252d_low_distance_pct')}%

=== PREDICTION (from senior analyst) ===
news_type          : {primary.get('news_type')}
already_priced_in  : {primary.get('already_priced_in')}
quantified_ratio   : {primary.get('quantified_ratio')}
short  : {h_short.get('direction')}/{h_short.get('magnitude')}  conf={h_short.get('confidence')}
medium : {h_medium.get('direction')}/{h_medium.get('magnitude')}  conf={h_medium.get('confidence')}
large  : {h_large.get('direction')}/{h_large.get('magnitude')}  conf={h_large.get('confidence')}

Per-company magnitude bands (anchored to ATR):
{json.dumps(db.get('magnitude_bands_pct') or {}, ensure_ascii=False, indent=2)}

=== YOUR TASK ===
Vote ONE of: "BUY", "HOLD", "SELL".
Also propose an entry price range (low_egp, high_egp) anchored on the last
close and distance to support/resistance. For BUY votes, set a range slightly
below the last close (use 20d low and ATR as guides). For SELL votes, set a
range slightly above. For HOLD, repeat the last close as both endpoints.

=== OUTPUT (JSON only, no markdown, no prose) ===
{{
  "action":            "BUY" | "HOLD" | "SELL",
  "entry_price_egp":   {{"low": <float>, "high": <float>}},
  "confidence":        <float 0.0-1.0>,
  "rationale":         "<1 short Arabic sentence, no jargon>"
}}"""


# =============================================================================
#  Single-voter call (Claude)
# =============================================================================

def _claude_call(prompt: str, temperature: float, max_tokens: int) -> dict:
    """One Claude HTTP call. Returns parsed JSON dict; raises on hard failures."""
    resp = requests.post(
        f"{CLAUDE_BASE_URL.rstrip('/')}/v1/messages",
        headers={
            "x-api-key":         CLAUDE_API_KEY,
            "anthropic-version": "2023-06-01",
            "Content-Type":      "application/json",
        },
        json={
            "model":       CLAUDE_MODEL,
            "messages":    [{"role": "user", "content": prompt}],
            "max_tokens":  max_tokens,
            "temperature": temperature,
        },
        timeout=VOTER_TIMEOUT_S,
    )
    resp.raise_for_status()
    data = resp.json()

    # Anthropic-native: first text block
    if isinstance(data, dict) and "content" in data and isinstance(data["content"], list):
        for block in data["content"]:
            if isinstance(block, dict) and block.get("type") == "text" and "text" in block:
                return _parse_json(block["text"])
        raise ValueError(f"No text block. stop_reason={data.get('stop_reason')}")
    # OpenAI-compatible proxy shape
    if isinstance(data, dict) and "choices" in data:
        msg = data["choices"][0].get("message", {})
        return _parse_json(msg.get("content", ""))
    raise ValueError(f"Unrecognised Claude response shape: keys={list(data) if isinstance(data, dict) else type(data).__name__}")


def vote_once(prompt: str, idx: int) -> dict:
    """Wrapper that captures latency + error info per voter."""
    t0 = time.perf_counter()
    try:
        parsed = _claude_call(prompt, temperature=VOTER_TEMP, max_tokens=VOTER_MAX_TOKENS)
        return {
            "voter":     idx,
            "ok":        True,
            "latency_s": round(time.perf_counter() - t0, 2),
            "vote":      parsed,
        }
    except Exception as exc:
        return {
            "voter":     idx,
            "ok":        False,
            "latency_s": round(time.perf_counter() - t0, 2),
            "error":     f"{type(exc).__name__}: {exc}",
        }


# =============================================================================
#  Aggregation
# =============================================================================

def aggregate_votes(votes: list[dict]) -> dict:
    """Majority on action; median on entry-price endpoints; mean on confidence."""
    successful = [v for v in votes if v.get("ok")]
    if not successful:
        return {
            "consensus_action":   None,
            "action_tally":       {},
            "entry_price_median": None,
            "mean_confidence":    None,
            "n_voters":           len(votes),
            "n_successful":       0,
        }

    tally: dict[str, int] = {"BUY": 0, "HOLD": 0, "SELL": 0}
    lows, highs, confs = [], [], []
    for v in successful:
        vote = v.get("vote") or {}
        action = vote.get("action")
        if action in tally:
            tally[action] += 1
        ep = vote.get("entry_price_egp") or {}
        if isinstance(ep.get("low"),  (int, float)): lows.append(float(ep["low"]))
        if isinstance(ep.get("high"), (int, float)): highs.append(float(ep["high"]))
        if isinstance(vote.get("confidence"), (int, float)):
            confs.append(float(vote["confidence"]))

    consensus = max(tally, key=lambda k: tally[k]) if any(tally.values()) else None
    return {
        "consensus_action":  consensus,
        "action_tally":      tally,
        "entry_price_median": {
            "low":  round(median(lows),  2) if lows  else None,
            "high": round(median(highs), 2) if highs else None,
        },
        "mean_confidence":   round(sum(confs) / len(confs), 3) if confs else None,
        "n_voters":          len(votes),
        "n_successful":      len(successful),
    }


# =============================================================================
#  Per-trace voting
# =============================================================================

def vote_on_trace(trace_path: Path, out_dir: Path) -> dict:
    with open(trace_path, "r", encoding="utf-8") as f:
        trace = json.load(f)

    prompt = _build_voter_prompt(trace)

    print(f"\n  {trace_path.name}  -> firing {N_VOTERS} voters in parallel (temp={VOTER_TEMP})...")
    t0 = time.perf_counter()
    votes: list[dict] = []
    with ThreadPoolExecutor(max_workers=N_VOTERS) as pool:
        futures = {pool.submit(vote_once, prompt, i + 1): i + 1 for i in range(N_VOTERS)}
        for fut in as_completed(futures):
            v = fut.result()
            tag = "OK " if v["ok"] else "ERR"
            print(f"    voter#{v['voter']}  {tag}  ({v['latency_s']}s)")
            votes.append(v)
    wall = round(time.perf_counter() - t0, 2)

    agg = aggregate_votes(votes)
    summary = {
        "trace_file":         trace_path.name,
        "ticker":             trace.get("ticker"),
        "article_date":       trace.get("article_date"),
        "wall_clock_s":       wall,
        "aggregate":          agg,
        "votes":              votes,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / trace_path.name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(
        f"    -> {agg['n_successful']}/{agg['n_voters']} successful   "
        f"action={agg['consensus_action']}  "
        f"entry={agg['entry_price_median']}   "
        f"wall={wall}s   saved -> {out_path}"
    )
    return summary


# =============================================================================
#  Main
# =============================================================================

def main():
    in_dir  = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)

    if not in_dir.exists():
        print(f"ERROR: input dir {in_dir} does not exist. Run pipeline3.py first.")
        sys.exit(1)

    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if args:
        targets = [in_dir / name for name in args if (in_dir / name).exists()]
        missing = [name for name in args if not (in_dir / name).exists()]
        for m in missing:
            print(f"WARN: trace not found: {m}")
    else:
        targets = sorted(in_dir.glob("*.json"))

    if not targets:
        print(f"No traces to vote on under {in_dir}.")
        return

    print("=" * 62)
    print("  EGX30 Voting Layer (experimental, multi-Claude, parallel)")
    print(f"  Input dir   : {in_dir}")
    print(f"  Output dir  : {out_dir}")
    print(f"  Voters      : {N_VOTERS}    Temperature: {VOTER_TEMP}")
    print(f"  Traces      : {len(targets)}")
    print(f"  Model       : {CLAUDE_MODEL}")
    print("=" * 62)

    for t in targets:
        try:
            vote_on_trace(t, out_dir)
        except Exception as exc:
            print(f"  [WARN] {t.name} failed: {type(exc).__name__}: {exc}")

    print("\nDone.")


if __name__ == "__main__":
    main()
