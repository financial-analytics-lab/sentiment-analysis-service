# pipeline.py
"""
EGX30 News-Impact Pipeline — orchestrates the three-agent system.

Flow for each article:
  1. Load context (price stats) for the trading day BEFORE the article date.
  2. Run Arabic sentiment model (CAMeLBERT).
  3. Call Impact Agent  (Groq Llama 3.3 70B) -> direction / magnitude / horizon.
  4. Call Spillover Agent (Groq Llama 3.3 70B) -> affected peers.
  5. Call Critic Agent  (Claude) -> final authoritative prediction.
  6. Save full trace to outputs/predictions/{ticker}_{date}_{hash}.json.
"""

# =============================================================================
#  TOP-LEVEL CONTROLS  -- edit these before each run
# =============================================================================

MAX_NEWS = 1
# Number of articles to process from all_news_by_date.json.
# Articles are taken in chronological order (earliest date first).
# Set to None to process all articles.

OUTPUT_DIR = "outputs/predictions"

# --- Model identifiers (must match what your API provider accepts) ------------
GROQ_MODEL   = "qwen/qwen3-32b"
CLAUDE_MODEL = "claude-sonnet-4.6"

# --- Request limits -----------------------------------------------------------
GROQ_MAX_TOKENS   = 1200
CLAUDE_MAX_TOKENS = 2000
MAX_RETRIES       = 5
RETRY_BACKOFF     = 2.0   # seconds; multiplied by attempt number on each retry
RATE_LIMIT_BACKOFF = 30.0  # base wait after 429; multiplied by attempt number

# =============================================================================

import json
import os
import time
import hashlib
from datetime import datetime
from pathlib import Path

import requests

from config import GROQ_API_KEY, CLAUDE_API_KEY, CLAUDE_BASE_URL
from features import compute_context, COMPANY_DESCRIPTIONS
from prompts import (
    impact_agent_prompt,
    spillover_agent_prompt,
    critic_agent_prompt,
    build_spillover_candidates,
)
from arabic_sentiment import ArabicSentimentAnalyzer

# Resolve paths relative to this file's location
_HERE      = Path(__file__).parent
NEWS_FILE  = _HERE / "output" / "all_news_by_date.json"
if not NEWS_FILE.exists():
    NEWS_FILE = _HERE.parent / "all_news_by_date.json"
PRICES_FILE = _HERE / "egyptian_stocks_2020_2025.json"


# =============================================================================
#  Data helpers
# =============================================================================

def _load_news() -> dict:
    with open(NEWS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_price_data() -> dict:
    """Load raw price JSON (used only for prev-trading-day lookup)."""
    with open(PRICES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _flatten_articles(news_data: dict) -> list:
    """Return all articles sorted chronologically (date order, then by position)."""
    articles = []
    for date in sorted(news_data["by_date"].keys()):
        articles.extend(news_data["by_date"][date])
    return articles


def _prev_trading_day(ticker: str, article_date: str, price_data: dict) -> str:
    """Last date with a recorded close for `ticker` that is strictly before `article_date`."""
    prices = price_data.get(ticker, {}).get("prices", [])
    candidates = [r["Date"] for r in prices if r["Date"] < article_date]
    if not candidates:
        raise ValueError(f"No trading day before {article_date} for ticker {ticker}.")
    return max(candidates)


# =============================================================================
#  API call wrappers
# =============================================================================

def _call_groq(prompt: str) -> str:
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model":      GROQ_MODEL,
            "messages":   [{"role": "user", "content": prompt}],
            "max_tokens": GROQ_MAX_TOKENS,
            "temperature": 0,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


_last_claude_thinking: str = ""   # module-level; pipeline reads this after each call


def _call_claude(prompt: str) -> str:
    resp = requests.post(
        f"{CLAUDE_BASE_URL.rstrip('/')}/v1/messages",
        headers={
            "x-api-key":         CLAUDE_API_KEY,
            "anthropic-version": "2023-06-01",
            "Content-Type":      "application/json",
        },
        json={
            "model":      CLAUDE_MODEL,
            "messages":   [{"role": "user", "content": prompt}],
            "max_tokens": CLAUDE_MAX_TOKENS,
            "temperature": 0,
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    # Anthropic-native: {"content": [{"type": "text", "text": "..."}], ...}
    # Extended thinking adds a {"type": "thinking", ...} block before the text block.
    if isinstance(data, dict) and "content" in data:
        global _last_claude_thinking
        content = data["content"]
        if isinstance(content, list) and content:
            # Capture thinking block if present
            for block in content:
                if isinstance(block, dict) and block.get("type") == "thinking":
                    _last_claude_thinking = block.get("thinking", "")
            # Return the first text block
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text" and "text" in block:
                    return block["text"]
            # Fallback: first block regardless of type
            first = content[0]
            if isinstance(first, dict) and "text" in first:
                return first["text"]
            return str(first)
        return str(content)
    # OpenAI-compat: {"choices": [{"message": {"content": "..."}}]}
    if isinstance(data, dict) and "choices" in data:
        msg = data["choices"][0]
        if isinstance(msg, dict) and "message" in msg:
            return msg["message"]["content"]
        return str(msg)
    # Direct string fields (some providers)
    for key in ("response", "result", "output", "message"):
        if isinstance(data, dict) and key in data:
            val = data[key]
            if isinstance(val, dict) and "content" in val:
                return val["content"]
            return str(val)
    raise ValueError(f"Unrecognised Claude response format. Keys: {list(data.keys())}")


# =============================================================================
#  JSON parsing + retry logic
# =============================================================================

def _parse_json(raw: str) -> dict:
    """
    Extract a JSON object from raw LLM output.
    Handles markdown fences, leading prose, and Python-style single-quoted dicts.
    """
    raw = raw.strip()

    # Strip markdown fences
    for fence in ("```json", "```"):
        if raw.startswith(fence):
            raw = raw[len(fence):]
    if raw.endswith("```"):
        raw = raw[:-3]
    raw = raw.strip()

    # Isolate first { ... } block
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1:
        raw = raw[start : end + 1]

    # 1st try: strict JSON
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 2nd try: Python literal (handles single-quoted keys/values from LLMs)
    try:
        import ast
        result = ast.literal_eval(raw)
        if isinstance(result, dict):
            # Round-trip through json to normalise types
            return json.loads(json.dumps(result, ensure_ascii=False))
    except (ValueError, SyntaxError):
        pass

    # Re-raise as JSONDecodeError so retry logic sees it
    return json.loads(raw)


def _call_with_retry(
    call_fn,
    prompt: str,
    agent_name: str,
) -> tuple:
    """
    Call an agent, parse JSON, retry up to MAX_RETRIES times.
    On a JSON parse failure, appends an error correction instruction
    to the prompt before the next attempt.

    Returns (parsed_dict, raw_string).
    """
    current_prompt = prompt
    last_exc = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw    = call_fn(current_prompt)
            parsed = _parse_json(raw)
            if attempt > 1:
                print(f"    [{agent_name}] OK on attempt {attempt}.")
            return parsed, raw

        except json.JSONDecodeError as exc:
            last_exc = exc
            print(f"    [{agent_name}] JSON error (attempt {attempt}): {exc}")
            print(f"    [{agent_name}] Raw response (first 300 chars): {repr(raw[:300])}")
            if attempt < MAX_RETRIES:
                current_prompt = (
                    current_prompt
                    + f"\n\nYour previous response was not valid JSON.\n"
                      f"Parse error: {exc}\n"
                      f"Output ONLY the JSON object -- no markdown, no surrounding text."
                )
                time.sleep(RETRY_BACKOFF * attempt)

        except requests.HTTPError as exc:
            last_exc = exc
            status = exc.response.status_code
            print(f"    [{agent_name}] HTTP {status} (attempt {attempt})")
            if attempt < MAX_RETRIES:
                if status == 429:
                    retry_after = exc.response.headers.get("Retry-After")
                    wait = float(retry_after) if retry_after else RATE_LIMIT_BACKOFF * attempt
                    print(f"    [{agent_name}] Rate limited — waiting {wait:.0f}s before retry...")
                    time.sleep(wait)
                else:
                    time.sleep(RETRY_BACKOFF * attempt)

        except Exception as exc:
            last_exc = exc
            print(f"    [{agent_name}] Error (attempt {attempt}): {type(exc).__name__}: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)

    raise RuntimeError(f"[{agent_name}] Failed after {MAX_RETRIES} attempts: {last_exc}")


# =============================================================================
#  Output
# =============================================================================

def _save_trace(trace: dict) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ticker = trace["ticker"]
    date   = trace["article_date"]
    h      = hashlib.md5(
        f"{ticker}{date}{trace['article'].get('title', '')}".encode()
    ).hexdigest()[:8]
    path = os.path.join(OUTPUT_DIR, f"{ticker}_{date}_{h}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trace, f, ensure_ascii=False, indent=2)
    return path


# =============================================================================
#  Main pipeline
# =============================================================================

def run(max_news: int = MAX_NEWS) -> list:
    """
    Process the first `max_news` articles from all_news_by_date.json.
    Returns a list of result summary dicts.
    """
    print("=" * 62)
    print("  EGX30 News-Impact Pipeline")
    print(f"  Articles to process : {max_news if max_news is not None else 'ALL'}")
    print(f"  Groq model          : {GROQ_MODEL}")
    print(f"  Claude model        : {CLAUDE_MODEL}")
    print("=" * 62)

    news_data   = _load_news()
    price_data  = _load_price_data()

    articles = _flatten_articles(news_data)
    if max_news is not None:
        articles = articles[:max_news]
    print(f"\nLoaded {len(articles)} article(s) from {NEWS_FILE.name}.\n")

    print("Loading Arabic sentiment model...")
    sentiment_model = ArabicSentimentAnalyzer()

    results: list[dict] = []

    for idx, article in enumerate(articles, start=1):
        ticker       = article.get("symbol", "")
        article_date = article.get("date",   "")
        title        = article.get("title",  "")

        print(f"\n{'─'*62}")
        print(f"  [{idx}/{len(articles)}]  {ticker}  |  {article_date}")
        print(f"  {title[:90]}")
        print(f"{'─'*62}")

        # ── guard: ticker must be in price data ───────────────────────────────
        if ticker not in price_data:
            print(f"  SKIP -- '{ticker}' has no price data.")
            continue

        # ── previous trading day (context cutoff) ─────────────────────────────
        try:
            context_date = _prev_trading_day(ticker, article_date, price_data)
        except ValueError as exc:
            print(f"  SKIP -- {exc}")
            continue

        print(f"  Context date (prev trading day) : {context_date}")

        # ── compute quantitative context ──────────────────────────────────────
        try:
            ctx = compute_context(ticker, context_date)
        except ValueError as exc:
            print(f"  SKIP -- context error: {exc}")
            continue

        # Override event_date so prompts show the actual news date, not cutoff.
        ctx["event_date"] = article_date

        # ── Arabic sentiment ──────────────────────────────────────────────────
        text      = (article.get("body") or article.get("title") or "").strip()
        sentiment = sentiment_model.analyze(text)
        print(f"  Sentiment : {sentiment['label']}  score={sentiment['score']:+.4f}  "
              f"probs={sentiment['class_probs']}")

        # ── spillover candidate universe ──────────────────────────────────────
        candidates = build_spillover_candidates(ctx, COMPANY_DESCRIPTIONS)

        # ── Impact Agent (Groq) ───────────────────────────────────────────────
        print("  Calling Impact Agent (Groq)...")
        p_impact = impact_agent_prompt(
            ctx, sentiment["score"], sentiment["label"], article
        )
        impact_parsed, impact_raw = _call_with_retry(_call_groq, p_impact, "ImpactAgent")
        print(f"    -> direction={impact_parsed.get('direction')}  "
              f"magnitude={impact_parsed.get('magnitude')}  "
              f"confidence={impact_parsed.get('confidence')}")

        # ── Spillover Agent (Groq) ────────────────────────────────────────────
        print("  Calling Spillover Agent (Groq)...")
        p_spillover = spillover_agent_prompt(ctx, article, candidates)
        spillover_parsed, spillover_raw = _call_with_retry(
            _call_groq, p_spillover, "SpilloverAgent"
        )
        n_spill = len(spillover_parsed.get("spillovers", []))
        print(f"    -> {n_spill} spillover(s) identified.")

        # ── Critic Agent (Claude) ─────────────────────────────────────────────
        print("  Calling Critic Agent (Claude)...")
        p_critic = critic_agent_prompt(
            ctx,
            sentiment["score"],
            sentiment["label"],
            article,
            impact_parsed,
            spillover_parsed,
        )
        critic_parsed, critic_raw = _call_with_retry(_call_claude, p_critic, "CriticAgent")
        final = critic_parsed.get("primary_company", {})
        print(f"    -> direction={final.get('direction')}  "
              f"magnitude={final.get('magnitude')}  "
              f"confidence={final.get('confidence')}")

        # ── save full trace ───────────────────────────────────────────────────
        trace = {
            "ticker":        ticker,
            "article_date":  article_date,
            "context_date":  context_date,
            "processed_at":  datetime.now().isoformat(),
            "article":       article,
            "sentiment":     sentiment,
            "context":       ctx,
            "prompts": {
                "impact":    p_impact,
                "spillover": p_spillover,
                "critic":    p_critic,
            },
            "impact_agent":    {"output": impact_parsed,    "raw": impact_raw},
            "spillover_agent": {"output": spillover_parsed,  "raw": spillover_raw},
            "critic_agent":    {"output": critic_parsed, "raw": critic_raw,
                                "thinking": _last_claude_thinking},
        }
        path = _save_trace(trace)
        print(f"  Saved -> {path}")

        results.append({
            "ticker":       ticker,
            "article_date": article_date,
            "context_date": context_date,
            "direction":    final.get("direction"),
            "magnitude":    final.get("magnitude"),
            "confidence":   final.get("confidence"),
            "path":         path,
        })

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  Done. {len(results)}/{len(articles)} article(s) processed successfully.")
    if results:
        print()
        for r in results:
            print(f"  {r['ticker']:<8} {r['article_date']}  "
                  f"{str(r['direction']):<8} {str(r['magnitude']):<8} "
                  f"conf={r['confidence']}")
    print("=" * 62)
    return results


if __name__ == "__main__":
    run()
