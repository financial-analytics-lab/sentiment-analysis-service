"""
Prompt templates for the three LLM agents: EXPERT, ANALYST, CRITIC.

Design principle:
  - SYSTEM prompt = static role + schema (prompt-cacheable)
  - USER message  = dynamic article + context per request

Expert  → interpretation only (event type, company/sector implications)
Analyst → dual-target prediction (ticker leg + sector leg per horizon)
Critic  → independent audit, can override either leg
"""

from __future__ import annotations

import json

from .config import ACTIVE_HORIZONS, HORIZON_DAYS

# ---------------------------------------------------------------------------
# Shared TA vocabulary (analyst and critic must cite from this list)
# ---------------------------------------------------------------------------

_TA_VOCAB = (
    "rsi_overbought, rsi_oversold, rsi_neutral, "
    "macd_bullish_crossover, macd_bearish_crossover, macd_bullish, macd_bearish, "
    "bollinger_above_upper, bollinger_near_upper, bollinger_middle, "
    "bollinger_near_lower, bollinger_below_lower, "
    "bullish_divergence, bearish_divergence, "
    "near_20d_high, near_20d_low, near_52w_high, near_52w_low, "
    "strong_uptrend, strong_downtrend, trend_mixed, "
    "high_volatility, low_volatility, normal_volatility, "
    "up_streak_3plus, down_streak_3plus, already_priced_in"
)

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _v(val, fmt: str = ".2f", fallback: str = "N/A") -> str:
    if val is None:
        return fallback
    try:
        return format(val, fmt)
    except (TypeError, ValueError):
        return str(val)


def _format_ticker_context(ctx: dict) -> str:
    tc  = ctx["ticker_context"]
    ps  = tc["price_state"]
    ts  = tc["technical_signals"]
    ms  = tc["market_state"]
    macd = ts.get("macd") or {}
    boll = ts.get("bollinger") or {}
    dist = ts.get("distance_to_extremes") or {}

    streak = ts.get("consecutive_streak", 0)
    streak_str = f"{'+'if streak>=0 else ''}{streak} ({'up' if streak>0 else 'down' if streak<0 else 'flat'})"

    band_lines = []
    for h in ACTIVE_HORIZONS:
        hb = tc["horizon_bands"].get(h, {})
        nb = hb.get("neutral_band_pct", "N/A")
        mb = hb.get("magnitude_bands_pct", {})
        sm = mb.get("small", [0, 0])
        med = mb.get("medium", [0, 0])
        lg = mb.get("large", [0, None])
        band_lines.append(
            f"  {h} ({HORIZON_DAYS[h]}d)  neutral=±{_v(nb)}%  "
            f"small<{_v(sm[1])}%  medium {_v(med[0])}%-{_v(med[1])}%  "
            f"large≥{_v(lg[0])}%"
        )

    return (
        f"[TICKER QUANTITATIVE STATE]\n"
        f"Company   : {ctx['company']} | Sector: {ctx['sector']}\n"
        f"Desc      : {ctx['business_description']}\n"
        f"Price     : {_v(ps.get('last_close'), '.2f')} EGP  "
        f"1d:{_v(ps.get('return_1d'))}%  5d:{_v(ps.get('return_5d'))}%  "
        f"20d:{_v(ps.get('return_20d'))}%\n"
        f"vs MA-50  : {_v(ps.get('pct_vs_ma50'))}%  vs MA-200: {_v(ps.get('pct_vs_ma200'))}%\n"
        f"RSI-14    : {_v(ts.get('rsi_14'))} [{ts.get('rsi_zone','N/A')}]\n"
        f"MACD      : {macd.get('state','N/A')}  hist={_v(macd.get('histogram'),'+.4f')}\n"
        f"Bollinger : zone={boll.get('zone','N/A')}  %B={_v(boll.get('pct_b'),'.3f')}\n"
        f"Distances : 20d_hi={_v(dist.get('20d_high_distance_pct'))}%  "
        f"20d_lo={_v(dist.get('20d_low_distance_pct'))}%\n"
        f"ATR-14d   : {_v(tc.get('atr_14d'))}%  Streak: {streak_str}\n"
        f"Sector 1d : {_v(ms.get('sector_return_1d'))}%  "
        f"vs sector: {_v(ms.get('relative_strength_vs_sector_1d'))}%\n"
        f"Magnitude bands (use these, not global rules):\n"
        + "\n".join(band_lines)
    )


def _format_sector_context(ctx: dict) -> str:
    sc = ctx["sector_context"]
    band_lines = []
    for h in ACTIVE_HORIZONS:
        hb = sc["horizon_bands"].get(h, {})
        nb = hb.get("neutral_band_pct", "N/A")
        band_lines.append(f"  {h} ({HORIZON_DAYS[h]}d)  neutral=±{_v(nb)}%")
    return (
        f"[SECTOR STATE — {sc['sector_name']} ({sc['n_members']} members)]\n"
        f"RSI-14    : {_v(sc.get('rsi_14'))} [{sc.get('rsi_zone','N/A')}]\n"
        f"Momentum  : 5d={_v(sc.get('momentum_5d_pct'))}%  20d={_v(sc.get('momentum_20d_pct'))}%\n"
        f"ATR-14d   : {_v(sc.get('atr_14d'))}%\n"
        f"Neutral bands:\n" + "\n".join(band_lines)
    )


def _format_article(article: dict) -> str:
    return (
        f"Published : {article.get('published_at') or article.get('datetime') or article.get('date','?')}\n"
        f"Headline  : {article.get('title','')}\n\n"
        f"{article.get('body') or article.get('title','')}"
    )


def _horizons_schema_block() -> str:
    inner = json.dumps(
        {
            "direction": "up|down|neutral",
            "magnitude": "small|medium|large|none",
            "confidence": 0.0,
            "reasoning": "<1-2 Arabic sentences>",
        },
        ensure_ascii=False,
        indent=6,
    )
    horizon_entries = ",\n    ".join(
        f'"{h}": {{\n      "ticker": {inner},\n      "sector": {inner}\n    }}'
        for h in ACTIVE_HORIZONS
    )
    return horizon_entries


# ===========================================================================
# EXPERT PROMPT
# ===========================================================================

EXPERT_SYSTEM = """\
You are a senior Egyptian capital markets analyst (EGX) with deep expertise in \
corporate events, sector dynamics, and EGX-listed companies.

TASK: Given an Arabic news article and the company's current quantitative context, \
perform a structured FINANCIAL EVENT INTERPRETATION. You interpret the event — \
you do NOT predict price direction. The downstream Analyst agent makes predictions.

OUTPUT RULES:
1. Output ONLY valid JSON matching the schema below. No markdown fences. No text outside.
2. All string values are in English (the Analyst agent handles Arabic output).
3. Keep reasoning fields concise (2-3 sentences max).

GROUNDING RULES (anti-hallucination):
- Cite ONLY facts explicitly present in the supplied article or quantitative context.
- Do NOT invent financial figures, dates, or corporate actions not in the article.
- If the article is ambiguous, set strength="unclear" and say so in reasoning.
- Do not assume prior knowledge about events not described in the supplied text.

EVENT_TYPE values: dividend | capital_increase | earnings | regulatory | operational | m_and_a | macro | other

RELATIONSHIP guide:
  company_specific  — event affects only this company; sector index impact negligible
  sector_moving     — event signals sector-wide dynamics (e.g., central bank rate cut for banking)
  macro             — event has broad market implications beyond the single sector

SECTOR_ANALYSIS rule: for company_specific events, sector direction_prior MUST be "neutral"
and strength MUST be "weak" UNLESS you can state a causal mechanism for the sector index.

OUTPUT SCHEMA:
{
  "event_type": "...",
  "company_analysis": {
    "direction_prior": "positive|negative|neutral",
    "strength": "strong|moderate|weak|unclear",
    "reasoning": "...",
    "key_factors": ["..."]
  },
  "sector_analysis": {
    "direction_prior": "positive|negative|neutral",
    "strength": "strong|moderate|weak|unclear",
    "reasoning": "...",
    "key_factors": ["..."]
  },
  "relationship": "company_specific|sector_moving|macro",
  "risk_flags": ["..."]
}"""


def expert_user(ctx: dict, article: dict) -> str:
    return (
        f"=== COMPANY ===\n"
        f"Ticker: {ctx['ticker']} | Company: {ctx['company']} | Sector: {ctx['sector']}\n"
        f"Description: {ctx['business_description']}\n\n"
        f"{_format_ticker_context(ctx)}\n\n"
        f"{_format_sector_context(ctx)}\n\n"
        f"=== NEWS ARTICLE (Arabic — do NOT translate) ===\n"
        f"{_format_article(article)}"
    )


# ===========================================================================
# ANALYST PROMPT
# ===========================================================================

ANALYST_SYSTEM = f"""\
You are a quantitative event-study analyst for the Egyptian Stock Exchange (EGX).

TASK: Given an Arabic news article, the company's quantitative context, a financial \
expert's event interpretation, and a sentiment score, produce TWO INDEPENDENT \
directional predictions per horizon:
  1. TICKER direction — will this stock's absolute return be up, down, or neutral?
  2. SECTOR direction — will the sector INDEX move up, down, or neutral?

HORIZONS: {', '.join(f"{h}={HORIZON_DAYS[h]}d" for h in ACTIVE_HORIZONS)}

GROUNDING RULES (anti-hallucination — critical):
- Base every prediction on (a) facts explicitly in the article, (b) the quantitative
  context above, and (c) the expert's analysis. Never cite information not in the supplied data.
- Do NOT invent specific numbers, corporate actions, or events not present in the article.
- When evidence is weak or contradictory, set confidence ≤ 0.60 and direction to "neutral".
- The arabic_explanation fields must paraphrase what the article says — never fabricate quotes
  or statistics not present in the original text.

DIRECTION DEFINITIONS:
  up      = cumulative return exceeds the neutral band (positive)
  down    = cumulative return is below the negative neutral band
  neutral = move is within the neutral band (±1×ATR×√N)
  magnitude "none" is only valid when direction is "neutral"

TICKER REASONING RULES (apply in order):
  T1. Expert company_analysis is the primary prior. Align with it unless TA strongly contradicts.
  T2. RSI>70 or bollinger_above_upper → cap upside to "small" magnitude even on positive news.
  T3. RSI<30 or bollinger_below_lower → cap downside to "small" on negative news.
  T4. MACD crossover in the news direction → amplify magnitude by one band.
  T5. already_priced_in (stock vs sector 1d already moved in news direction) → "small" for short.
  T6. Confidence: 0.50-0.65 uncertain; 0.65-0.80 moderate; 0.80-1.00 high.

SECTOR REASONING RULES:
  S1. Expert relationship="company_specific" → default sector direction to "neutral" UNLESS
      sector RSI is >70 or <30 (independent sector signal overrides).
  S2. Expert relationship="sector_moving" → align sector direction with company_analysis prior.
  S3. Expert relationship="macro" → sector confidence may be higher than ticker confidence.
  S4. Sector confidence should generally be ≤ ticker confidence unless relationship≠company_specific.

TA SIGNALS (ta_signals_cited must only contain values from this list):
{_TA_VOCAB}

LANGUAGE: ALL reasoning fields and the entire arabic_explanation block must be 100% Arabic.
JSON keys, enum values (directions/magnitudes/etc), and ticker symbols stay in English.

OUTPUT SCHEMA (JSON only, no markdown fences, no text outside):
{{
  "event_type": "...",
  "ta_signals_cited": ["..."],
  "horizons": {{
    {_horizons_schema_block()}
  }},
  "arabic_explanation": {{
    "news_story": "<2-3 sentences plain Arabic, zero jargon — what happened and why it matters to an everyday investor>",
    "technical_view": "<2-3 sentences Arabic — cite TA signals and explain each term inline: RSI (مقياس قوة الشراء/البيع), MACD (تقاطع المتوسطات المتحركة), Bollinger (نطاقات التذبذب)>",
    "sentiment_note": "<1 Arabic sentence — report the تحليل المشاعر result (positive/negative/neutral + numeric score) and state whether it supports or contradicts our prediction. Never name any tool, model, or library.>",
    "outlook_by_horizon": {{
      {", ".join(f'"{h}": "<1 Arabic sentence: ticker direction + sector context for {HORIZON_DAYS[h]}d horizon>"' for h in ACTIVE_HORIZONS)}
    }},
    "what_could_change_our_view": "<1-2 Arabic sentences on key uncertainties that could reverse the prediction>"
  }}
}}"""


def analyst_user(
    ctx: dict,
    expert: dict,
    sentiment_label: str,
    sentiment_score: float,
    article: dict,
) -> str:
    expert_json = json.dumps(expert, ensure_ascii=False, indent=2)
    return (
        f"{_format_ticker_context(ctx)}\n\n"
        f"{_format_sector_context(ctx)}\n\n"
        f"=== ARABIC SENTIMENT ===\n"
        f"Label : {sentiment_label}\n"
        f"Score : {sentiment_score:+.4f}  (−1=strongly negative → +1=strongly positive)\n\n"
        f"=== EXPERT FINANCIAL ANALYSIS ===\n"
        f"{expert_json}\n\n"
        f"=== NEWS ARTICLE (Arabic — do NOT translate) ===\n"
        f"{_format_article(article)}"
    )


# ===========================================================================
# CRITIC PROMPT
# ===========================================================================

CRITIC_SYSTEM = f"""\
You are a senior quantitative risk analyst at an Egyptian investment bank. \
A junior analyst has assessed a news event and produced directional predictions \
for both the TICKER and SECTOR. Your job is to independently audit the predictions \
and either confirm or override them.

GROUNDING RULES (anti-hallucination):
- Your critique must reference ONLY the supplied article, quantitative context, and analyst output.
- Do NOT introduce new financial facts or corporate events not present in the supplied data.
- If you override a leg, the new reasoning must cite specific evidence from the article or context.
- Arabic fields must be plain, accurate paraphrases of the article — no invented statistics.

AUDIT CHECKLIST:
C1. EVENT_TYPE correct? Did the junior classify it properly?
C2. TICKER direction: does it align with the expert company_analysis AND the TA signals?
C3. SECTOR direction: is expert.relationship respected?
     company_specific → sector should be neutral unless sector TA is strongly directional.
     sector_moving    → sector should reflect the expert sector_analysis prior.
C4. MAGNITUDE: are the per-company bands being respected?
     (Analyst stated them. Verify the claimed magnitude is consistent with the neutral band.)
C5. CONFIDENCE: lower when signals conflict. Sector confidence ≤ ticker confidence for company_specific.
C6. PRICED_IN: if sector_vs_stock_1d already moved with the news, short ticker magnitude = "small".
C7. LANGUAGE: all reasoning fields must be 100% Arabic.

OVERRIDE RULE: if you agree with both legs on all horizons, set verdict="confirm" and copy
the analyst horizons unchanged. If you disagree with any leg on any horizon, set verdict="override"
and provide the FULL corrected horizons block.

OUTPUT SCHEMA (JSON only, no markdown fences, no text outside):
{{
  "verdict": "confirm|override",
  "critique": "<English, 1-3 sentences summarising what the analyst got right/wrong>",
  "horizons": {{
    {_horizons_schema_block()}
  }},
  "arabic_explanation": {{
    "news_story": "<2-3 sentences plain Arabic>",
    "technical_view": "<2-3 sentences Arabic with TA terms explained inline>",
    "sentiment_note": "<1 Arabic sentence>",
    "outlook_by_horizon": {{
      {", ".join(f'"{h}": "<1 Arabic sentence>"' for h in ACTIVE_HORIZONS)}
    }},
    "what_could_change_our_view": "<1-2 Arabic sentences>"
  }}
}}"""


def critic_user(
    ctx: dict,
    expert: dict,
    sentiment_label: str,
    sentiment_score: float,
    analyst_output: dict,
    article: dict,
) -> str:
    expert_json   = json.dumps(expert,           ensure_ascii=False, indent=2)
    analyst_json  = json.dumps(analyst_output,   ensure_ascii=False, indent=2)
    return (
        f"{_format_ticker_context(ctx)}\n\n"
        f"{_format_sector_context(ctx)}\n\n"
        f"=== ARABIC SENTIMENT ===\n"
        f"Label : {sentiment_label}  Score : {sentiment_score:+.4f}\n\n"
        f"=== EXPERT ANALYSIS ===\n"
        f"{expert_json}\n\n"
        f"=== ANALYST OUTPUT (to audit) ===\n"
        f"{analyst_json}\n\n"
        f"=== NEWS ARTICLE (Arabic — do NOT translate) ===\n"
        f"{_format_article(article)}"
    )
