# pipeline3.py
"""
EGX30 News-Impact Pipeline (two-agent, TA-grounded, v2).

Flow per article:
  1. Load context + technical signals (no lookahead) for the trading day before the article.
  2. Run Arabic sentiment model (CAMeLBERT).
  3. Impact Agent (Groq Qwen3-32B) -> per-horizon prediction (short, medium) grounded
                                      in TA signals from the closed vocab.
  4. Critic Agent (Claude)         -> audits TA citations, finalizes prediction,
                                      writes plain-Arabic user explanation.
  5. Save full trace to outputs/predictions_v2/{ticker}_{date}_{hash}.json.
"""

# =============================================================================
#  TOP-LEVEL CONTROLS  -- edit these in one place
# =============================================================================

MAX_NEWS = 15
# Number of articles to process. Set to None for all.

OUTPUT_DIR = "outputs/predictions_v3"

# --- Models  (one knob per stage) ---------------------------------------------
GROQ_MODEL          = "qwen/qwen3-32b"      # ImpactAgent  (Groq)
CLAUDE_MODEL        = "claude-sonnet-4-6"   # CriticAgent  (Claude via proxy)
ARABIC_QA_MODEL     = "claude-sonnet-4-6"   # Arabic-QA    (Claude via proxy)

# --- Toggles ------------------------------------------------------------------
ARABIC_QA_ENABLED   = True   # 4th-stage Arabic editor (Claude). +5-15s, better Arabic.

# --- Token budgets ------------------------------------------------------------
GROQ_MAX_TOKENS     = 1500
CLAUDE_MAX_TOKENS   = 6000   # lowered from 8000 -- explanation_for_user is leaner now
QA_MAX_TOKENS       = 1500   # QA output is small (just the cleaned explanation dict)

# --- Retry / rate-limit -------------------------------------------------------
MAX_RETRIES         = 5
RETRY_BACKOFF       = 2.0    # seconds; multiplied by attempt number
RATE_LIMIT_BACKOFF  = 30.0   # base wait after 429; multiplied by attempt number

# =============================================================================

import json
import os
import time
import hashlib
from datetime import datetime
from pathlib import Path

import requests

from config import GROQ_API_KEY, CLAUDE_API_KEY, CLAUDE_BASE_URL
from features import (
    compute_context,
    news_in_trading_hours,
    HORIZON_TRADING_DAYS,   # single source of truth for horizons
)
from prompts import (
    HORIZONS,
    impact_agent_prompt,
    critic_agent_prompt,
    arabic_qa_prompt,
)
from arabic_sentiment import ArabicSentimentAnalyzer

# Resolve paths relative to this file's location
_HERE      = Path(__file__).parent
NEWS_FILE  = _HERE / "output" / "all_news_by_date_mubasher.json"
if not NEWS_FILE.exists():
    NEWS_FILE = _HERE.parent / "all_news_by_date_mubasher.json"
PRICES_FILE = _HERE / "egyptian_stocks_2020_2025.json"

_sentiment_model: ArabicSentimentAnalyzer | None = None


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
    """Return all articles sorted chronologically across supported news schemas."""
    symbol = (news_data.get("metadata") or {}).get("symbol")

    if isinstance(news_data.get("by_date"), dict):
        articles = []
        for date in sorted(news_data["by_date"].keys()):
            for article in news_data["by_date"][date]:
                if symbol and "symbol" not in article:
                    article = {**article, "symbol": symbol}
                articles.append(article)
        return articles

    if isinstance(news_data.get("articles"), list):
        articles = []
        for article in news_data["articles"]:
            if symbol and "symbol" not in article:
                article = {**article, "symbol": symbol}
            articles.append(article)

        articles.sort(
            key=lambda a: (
                a.get("date", ""),
                a.get("time", ""),
                a.get("datetime", ""),
                str(a.get("id", "")),
            )
        )
        return articles

    raise ValueError(
        "Unsupported news schema: expected either 'by_date' dict or 'articles' list."
    )


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
    # reasoning_format="hidden" tells Qwen3 to keep its internal reasoning but
    # NOT emit <think>...</think> in the response. Without it, the model wraps
    # its chain-of-thought around the JSON, which corrupts naive {...} slicing.
    return _call_groq_with_model(
        prompt, model=GROQ_MODEL, max_tokens=GROQ_MAX_TOKENS,
    )


def _call_groq_with_model(prompt: str, model: str, max_tokens: int) -> str:
    """Reusable Groq call -- lets us pick a different model per stage."""
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model":            model,
            "messages":         [{"role": "user", "content": prompt}],
            "max_tokens":       max_tokens,
            "temperature":      0,
            "reasoning_format": "hidden",
            "response_format":  {"type": "json_object"},
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# Module-level capture of the most recent Claude thinking block.
# Reset at the start of every _call_claude call so it never carries over.
_last_claude_thinking: str = ""


def _call_claude(prompt: str) -> str:
    """Default Claude call -- uses CLAUDE_MODEL + CLAUDE_MAX_TOKENS (Critic stage)."""
    return _call_claude_with_model(prompt, model=CLAUDE_MODEL, max_tokens=CLAUDE_MAX_TOKENS)


def _call_claude_with_model(prompt: str, model: str, max_tokens: int) -> str:
    """
    Parameterised Claude / proxy call. Lets each stage pick its own model
    and token budget. Returns the assistant's text output.

    Handles response shapes:
      - Anthropic-native: content blocks of type "thinking" and "text"
      - OpenAI-compatible proxies:  choices[0].message.content
      - Other proxy variants:       response/result/output/message at top level

    If the response is Anthropic-native but contains NO text block (typically
    because max_tokens was consumed entirely by the thinking budget), raises
    a clear ValueError rather than silently returning a stringified thinking
    block.
    """
    global _last_claude_thinking
    _last_claude_thinking = ""   # reset per call -- never carry stale thinking forward

    resp = requests.post(
        f"{CLAUDE_BASE_URL.rstrip('/')}/v1/messages",
        headers={
            "x-api-key":         CLAUDE_API_KEY,
            "anthropic-version": "2023-06-01",
            "Content-Type":      "application/json",
        },
        json={
            "model":      model,
            "messages":   [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
        },
        timeout=240,   # raised for slow proxies (api.claudy.cloud occasionally stalls)
    )
    resp.raise_for_status()
    data = resp.json()

    # ── Optional: log token usage if the provider reports it ──────────────────
    usage = data.get("usage") if isinstance(data, dict) else None
    if isinstance(usage, dict):
        print(
            f"    [Claude:{model}] tokens: "
            f"input={usage.get('input_tokens')}  "
            f"output={usage.get('output_tokens')}  "
            f"max={max_tokens}"
        )

    # ── Anthropic-native shape ────────────────────────────────────────────────
    if isinstance(data, dict) and "content" in data:
        content = data["content"]
        if isinstance(content, list) and content:
            for block in content:
                if isinstance(block, dict) and block.get("type") == "thinking":
                    _last_claude_thinking = block.get("thinking", "")

            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "text"
                    and "text" in block
                ):
                    return block["text"]

            block_types = [b.get("type") for b in content if isinstance(b, dict)]
            stop_reason = data.get("stop_reason")
            raise ValueError(
                f"Claude returned no text block. "
                f"Block types: {block_types}. "
                f"stop_reason: {stop_reason}. "
                f"Likely cause: max_tokens={max_tokens} for model={model} is too small "
                f"and the response was cut off mid-thinking. "
                f"Raise the token budget and re-run."
            )
        return str(content)

    # ── OpenAI-compatible shape (some proxies) ────────────────────────────────
    if isinstance(data, dict) and "choices" in data:
        msg = data["choices"][0]
        if isinstance(msg, dict) and "message" in msg:
            return msg["message"]["content"]
        return str(msg)

    # ── Other proxy variants ──────────────────────────────────────────────────
    for key in ("response", "result", "output", "message"):
        if isinstance(data, dict) and key in data:
            val = data[key]
            if isinstance(val, dict) and "content" in val:
                return val["content"]
            return str(val)

    raise ValueError(
        f"Unrecognised Claude response format. Top-level keys: "
        f"{list(data.keys()) if isinstance(data, dict) else type(data).__name__}"
    )


# =============================================================================
#  JSON parsing + retry logic
# =============================================================================

import re

_THINK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)

# Optional lenient fallback for LLM-produced JSON (unescaped quotes, trailing
# commas, etc.). Install with: pip install json-repair
try:
    from json_repair import repair_json as _repair_json  # type: ignore
    _HAS_JSON_REPAIR = True
except ImportError:
    _HAS_JSON_REPAIR = False


def _parse_json(raw: str):
    """
    Extract a JSON object from raw LLM output.
    Handles markdown fences, Qwen3 <think>...</think> reasoning blocks,
    leading prose, Python-style single-quoted dicts, and (if json-repair is
    installed) common LLM JSON malformations.
    """
    raw = raw.strip()

    # Strip Qwen3-style reasoning blocks first -- their curly braces would
    # otherwise corrupt the first-{ ... last-} slice below.
    raw = _THINK_RE.sub("", raw).strip()

    # Also handle an unclosed <think> at the start (max_tokens cut it off).
    if raw.lower().startswith("<think"):
        cut = raw.find("</think>")
        raw = raw[cut + len("</think>"):].strip() if cut != -1 else raw[raw.find("{"):]

    for fence in ("```json", "```"):
        if raw.startswith(fence):
            raw = raw[len(fence):]
    if raw.endswith("```"):
        raw = raw[:-3]
    raw = raw.strip()

    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1:
        raw = raw[start : end + 1]

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Fallback 1: Python literal (single-quoted dicts).
    try:
        import ast
        result = ast.literal_eval(raw)
        if isinstance(result, dict):
            return json.loads(json.dumps(result, ensure_ascii=False))
    except (ValueError, SyntaxError):
        pass

    # Fallback 2: json-repair handles unescaped quotes, trailing commas,
    # invalid control characters inside strings -- the usual LLM JSON sins.
    if _HAS_JSON_REPAIR:
        try:
            repaired = _repair_json(raw, return_objects=False)
            return json.loads(repaired)
        except Exception:
            pass

    return json.loads(raw)  # final attempt -- re-raises with original error


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
    raw = ""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw    = call_fn(current_prompt)
            parsed = _parse_json(raw)

            # Some models occasionally return a JSON array even when asked for
            # an object. Accept the common single-item wrapper and normalize it.
            if isinstance(parsed, list):
                if len(parsed) == 1 and isinstance(parsed[0], dict):
                    parsed = parsed[0]
                    print(f"    [{agent_name}] Normalized single-item JSON array to object.")
                else:
                    raise ValueError(
                        f"Expected top-level JSON object, got array(len={len(parsed)})."
                    )

            if not isinstance(parsed, dict):
                raise ValueError(
                    f"Expected top-level JSON object, got {type(parsed).__name__}."
                )

            if attempt > 1:
                print(f"    [{agent_name}] OK on attempt {attempt}.")
            return parsed, raw

        except json.JSONDecodeError as exc:
            last_exc = exc
            pos = getattr(exc, "pos", None)
            print(f"    [{agent_name}] JSON error (attempt {attempt}): {exc}")
            if pos is not None and 0 <= pos <= len(raw):
                lo, hi = max(0, pos - 120), min(len(raw), pos + 120)
                print(f"    [{agent_name}] Context around char {pos}: "
                      f"{repr(raw[lo:pos])}  >>HERE>>  {repr(raw[pos:hi])}")
            else:
                print(f"    [{agent_name}] Raw response (first 300 chars): {repr(raw[:300])}")
            if attempt < MAX_RETRIES:
                current_prompt = (
                    current_prompt
                    + f"\n\nYour previous response was not valid JSON.\n"
                      f"Parse error: {exc}\n"
                      f"Output ONLY the JSON object -- no markdown, no surrounding text."
                )
                time.sleep(RETRY_BACKOFF * attempt)

        except ValueError as exc:
            last_exc = exc
            print(f"    [{agent_name}] Payload-shape error (attempt {attempt}): {exc}")
            if attempt < MAX_RETRIES:
                current_prompt = (
                    current_prompt
                    + "\n\nYour previous response had the wrong JSON shape.\n"
                      "Return a single top-level JSON object (not an array/list).\n"
                      "Do not add markdown or extra explanatory text."
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
#  Output formatting helpers
# =============================================================================

def _fmt_per_horizon(per_horizon: dict | None) -> str:
    """Render a per-horizon dict as a compact, aligned multi-line summary."""
    if not isinstance(per_horizon, dict):
        return "    (no per_horizon payload)"
    lines = []
    for h in HORIZONS:
        row = per_horizon.get(h) or {}
        lines.append(
            f"      {h:<5} -> dir={str(row.get('direction')):<7} "
            f"mag={str(row.get('magnitude')):<6} "
            f"conf={row.get('confidence')}"
        )
    return "\n".join(lines)


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


def _get_sentiment_model() -> ArabicSentimentAnalyzer:
    global _sentiment_model
    if _sentiment_model is None:
        _sentiment_model = ArabicSentimentAnalyzer()
    return _sentiment_model


def _run_arabic_qa(explanation: dict) -> dict:
    """
    4th-stage Arabic editor pass over the critic's `explanation_for_user`.
    Uses Claude (same vendor as the critic) so Arabic-language quality is
    consistent across both stages. Adds ~5-15s through the proxy; the QA
    prompt is small so output is fast once the connection lands.

    Best-effort: on any failure (proxy down, JSON parse, etc.) we keep
    the original explanation unchanged rather than blocking the pipeline.
    """
    if not ARABIC_QA_ENABLED or not isinstance(explanation, dict) or not explanation:
        return explanation

    prompt = arabic_qa_prompt(explanation)
    try:
        raw = _call_claude_with_model(
            prompt, model=ARABIC_QA_MODEL, max_tokens=QA_MAX_TOKENS,
        )
        cleaned = _parse_json(raw)
        if isinstance(cleaned, dict) and cleaned.keys() & explanation.keys():
            print(f"    [ArabicQA] OK -- polished {len(cleaned)} field(s).")
            return cleaned
        print(f"    [ArabicQA] returned wrong shape; using original.")
    except Exception as exc:
        print(f"    [ArabicQA] failed ({type(exc).__name__}: {exc}); using original.")
    return explanation


def _build_news_timing(article: dict) -> dict:
    """
    Resolve the news timing (drives the evaluation window's START day):
      in_trading_hours = True  -> window starts at event_date itself
      in_trading_hours = False -> window starts at next trading day

    Returned dict is attached to ctx so both the prompts and the saved trace
    agree on the start-day convention.
    """
    in_hours   = news_in_trading_hours(article)
    arrived_at = article.get("datetime") or (
        f"{article.get('date', '?')} {article.get('time', '?')}"
    )
    return {
        "in_trading_hours": in_hours,
        "arrived_at_str":   arrived_at,
        "start_day_label": (
            "event_date (article day; news arrived during EGX hours)"
            if in_hours
            else "next trading day after event_date (news arrived outside EGX hours)"
        ),
    }


# =============================================================================
#  Single-article entry point (used by the FastAPI service if rewired)
# =============================================================================

def analyze_article(article: dict, *, save_trace: bool = True) -> dict:
    """Run the full pipeline for a single news article and return the trace."""
    price_data = _load_price_data()

    ticker = article.get("symbol", "")
    article_date = article.get("date", "")

    if not ticker:
        raise ValueError("Article is missing required field: symbol")
    if not article_date:
        raise ValueError("Article is missing required field: date")
    if ticker not in price_data:
        raise ValueError(f"Ticker '{ticker}' has no price data.")

    context_date = _prev_trading_day(ticker, article_date, price_data)
    ctx = compute_context(ticker, context_date)
    ctx["event_date"]  = article_date
    ctx["news_timing"] = _build_news_timing(article)

    sentiment_model = _get_sentiment_model()
    text = (article.get("body") or article.get("title") or "").strip()
    sentiment = sentiment_model.analyze(text)

    p_impact = impact_agent_prompt(ctx, sentiment["score"], sentiment["label"], article)
    impact_parsed, impact_raw = _call_with_retry(_call_groq, p_impact, "ImpactAgent")

    p_critic = critic_agent_prompt(
        ctx,
        sentiment["score"],
        sentiment["label"],
        article,
        impact_parsed,
    )
    critic_parsed, critic_raw = _call_with_retry(_call_claude, p_critic, "CriticAgent")
    critic_thinking = _last_claude_thinking

    # 4th stage: Arabic-QA polish on the user-facing block (best-effort).
    explanation_raw   = critic_parsed.get("explanation_for_user") or {}
    explanation_clean = _run_arabic_qa(explanation_raw)
    if isinstance(explanation_clean, dict):
        critic_parsed["explanation_for_user"] = explanation_clean

    trace = {
        "ticker":       ticker,
        "article_date": article_date,
        "context_date": context_date,
        "article":      article,
        "context":      ctx,
        "sentiment":    sentiment,
        "impact_agent": {"output": impact_parsed, "raw": impact_raw},
        "critic_agent": {
            "output":   critic_parsed,
            "raw":      critic_raw,
            "thinking": critic_thinking,
        },
        "arabic_qa": {
            "enabled":          ARABIC_QA_ENABLED,
            "model":            ARABIC_QA_MODEL,
            "explanation_pre":  explanation_raw,
            "explanation_post": explanation_clean,
        },
    }

    trace["final"] = critic_parsed.get("primary_company", {})

    if save_trace:
        trace["saved_to"] = _save_trace(trace)

    return trace


# =============================================================================
#  Main batch pipeline
# =============================================================================

def run(max_news: int = MAX_NEWS) -> list:
    """
    Process the first `max_news` articles from the source news file.
    Returns a list of result summary dicts.
    """
    print("=" * 62)
    print("  EGX30 News-Impact Pipeline (v2.2: Impact -> Critic -> Arabic-QA)")
    print(f"  Articles to process  : {max_news if max_news is not None else 'ALL'}")
    print(f"  Impact   (Groq)      : {GROQ_MODEL}")
    print(f"  Critic   (Claude)    : {CLAUDE_MODEL}")
    print(f"  Arabic-QA (Claude)   : {ARABIC_QA_MODEL}  (enabled={ARABIC_QA_ENABLED})")
    print(f"  Token budgets        : impact={GROQ_MAX_TOKENS}  critic={CLAUDE_MAX_TOKENS}  qa={QA_MAX_TOKENS}")
    print(f"  Horizons             : "
          + ", ".join(f"{h}({HORIZON_TRADING_DAYS[h]}d)" for h in HORIZONS))
    print("=" * 62)

    news_data   = _load_news()
    price_data  = _load_price_data()

    articles = _flatten_articles(news_data)
    if max_news is not None:
        articles = articles[:max_news]
    print(f"\nLoaded {len(articles)} article(s) from {NEWS_FILE.name}.\n")

    print("Loading Arabic sentiment model...")
    sentiment_model = _get_sentiment_model()

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
        ctx["event_date"]  = article_date
        ctx["news_timing"] = _build_news_timing(article)
        print(f"  News timing: {ctx['news_timing']['arrived_at_str']}  "
              f"in_hours={ctx['news_timing']['in_trading_hours']}  "
              f"-> {ctx['news_timing']['start_day_label']}")

        # ── Arabic sentiment ──────────────────────────────────────────────────
        text      = (article.get("body") or article.get("title") or "").strip()
        sentiment = sentiment_model.analyze(text)
        print(f"  Sentiment : {sentiment['label']}  score={sentiment['score']:+.4f}  "
              f"probs={sentiment['class_probs']}")

        # ── Impact Agent (Groq) ───────────────────────────────────────────────
        print("  Calling Impact Agent (Groq)...")
        p_impact = impact_agent_prompt(
            ctx, sentiment["score"], sentiment["label"], article
        )
        impact_parsed, impact_raw = _call_with_retry(_call_groq, p_impact, "ImpactAgent")
        print(f"    news_type={impact_parsed.get('news_type')}  "
              f"priced_in={impact_parsed.get('already_priced_in')}  "
              f"signals={impact_parsed.get('ta_signals_cited')}")
        print("    per_horizon:")
        print(_fmt_per_horizon(impact_parsed.get("per_horizon")))

        # ── Critic Agent (Claude) ─────────────────────────────────────────────
        print("  Calling Critic Agent (Claude)...")
        p_critic = critic_agent_prompt(
            ctx,
            sentiment["score"],
            sentiment["label"],
            article,
            impact_parsed,
        )
        critic_parsed, critic_raw = _call_with_retry(_call_claude, p_critic, "CriticAgent")
        critic_thinking = _last_claude_thinking

        primary = critic_parsed.get("primary_company", {}) or {}
        print(f"    news_type={primary.get('news_type')}  "
              f"priced_in={primary.get('already_priced_in')}  "
              f"signals={primary.get('ta_signals_cited')}")
        print("    per_horizon (final):")
        print(_fmt_per_horizon(primary.get("per_horizon")))

        # ── 4th stage: Arabic-QA polish ───────────────────────────────────────
        explanation_raw   = critic_parsed.get("explanation_for_user") or {}
        if ARABIC_QA_ENABLED:
            print("  Calling Arabic-QA Agent (Claude)...")
            explanation_clean = _run_arabic_qa(explanation_raw)
            if isinstance(explanation_clean, dict):
                critic_parsed["explanation_for_user"] = explanation_clean
        else:
            explanation_clean = explanation_raw

        # ── save full trace ───────────────────────────────────────────────────
        trace = {
            "ticker":       ticker,
            "article_date": article_date,
            "context_date": context_date,
            "processed_at": datetime.now().isoformat(),
            "article":      article,
            "sentiment":    sentiment,
            "context":      ctx,
            "prompts": {
                "impact": p_impact,
                "critic": p_critic,
            },
            "impact_agent": {"output": impact_parsed, "raw": impact_raw},
            "critic_agent": {
                "output":   critic_parsed,
                "raw":      critic_raw,
                "thinking": critic_thinking,
            },
            "arabic_qa": {
                "enabled":          ARABIC_QA_ENABLED,
                "model":            ARABIC_QA_MODEL,
                "explanation_pre":  explanation_raw,
                "explanation_post": explanation_clean,
            },
            "final": primary,
        }
        path = _save_trace(trace)
        print(f"  Saved -> {path}")

        # Build a flat summary row (one record per article).
        per_h = primary.get("per_horizon") or {}
        results.append({
            "ticker":            ticker,
            "article_date":      article_date,
            "context_date":      context_date,
            "news_type":         primary.get("news_type"),
            "already_priced_in": primary.get("already_priced_in"),
            "ta_signals_cited":  primary.get("ta_signals_cited") or [],
            "per_horizon":       {h: per_h.get(h) for h in HORIZONS},
            "path":              path,
        })

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  Done. {len(results)}/{len(articles)} article(s) processed successfully.")
    if results:
        print()
        col_w = 18
        header = (
            f"  {'TICKER':<7} {'DATE':<11} {'NEWS_TYPE':<18} "
            + " ".join(f"{h.upper():<{col_w}}" for h in HORIZONS)
        )
        print(header)
        print("  " + "-" * (len(header) - 2))

        def _cell(h):
            if not isinstance(h, dict):
                return "-"
            return f"{str(h.get('direction'))[:4]}/{str(h.get('magnitude'))[:3]}/{h.get('confidence')}"

        for r in results:
            per_h = r["per_horizon"]
            cells = " ".join(f"{_cell(per_h.get(h)):<{col_w}}" for h in HORIZONS)
            print(
                f"  {r['ticker']:<7} {r['article_date']:<11} "
                f"{str(r['news_type'])[:18]:<18} {cells}"
            )
    print("=" * 62)
    return results


if __name__ == "__main__":
    run()
