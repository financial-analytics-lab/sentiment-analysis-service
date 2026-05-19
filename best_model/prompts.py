# prompts.py
"""
Prompt templates for the two-agent EGX30 news-impact pipeline (TA-grounded).

    impact_agent_prompt() -> str   for Groq (Qwen3-32B or similar)
    critic_agent_prompt() -> str   for Claude (Anthropic)

HORIZON SCHEME (two buckets, strictly forward from t+1):
    short  -> t+1 to t+2  trading days  (neutral band +-1.5%)
    medium -> t+1 to t+20 trading days  (neutral band +-4%)

Magnitude bands scale with horizon (|cumulative abnormal return|):
                 small        medium         large
    short:       <1.5%        1.5% to <4%   >=4%
    medium:      <4%          4%   to <8%   >=8%

The agents predict {direction, magnitude, confidence} for BOTH horizons,
grounded in a fixed-vocabulary list of technical signals (`ta_signals_cited`)
that they must justify against the live quantitative context.
"""

import json


# -----------------------------------------------------------------------------
#  Horizon configuration  (single source of truth -- evaluate.py mirrors this)
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

HORIZON_LABEL_AR = {
    "short":  "المدى القصير (اليومان القادمان)",
    "medium": "المدى المتوسط (شهر تداول)",
}

# Per-horizon magnitude bands (|cumulative AR|, in %).
MAGNITUDE_BANDS_PCT = {
    "short":  {"small": (0.0, 1.5), "medium": (1.5, 4.0), "large": (4.0, float("inf"))},
    "medium": {"small": (0.0, 4.0), "medium": (4.0, 8.0), "large": (8.0, float("inf"))},
}


# -----------------------------------------------------------------------------
#  Closed TA-signal vocabulary  (the model MUST pick from this list)
# -----------------------------------------------------------------------------

TA_SIGNAL_VOCAB = [
    # Momentum / oscillators
    "rsi_overbought",            # rsi_14 > 70
    "rsi_oversold",              # rsi_14 < 30
    "rsi_neutral",
    "macd_bullish_crossover",    # crossed up today
    "macd_bearish_crossover",    # crossed down today
    "macd_bullish",              # MACD > signal (no fresh crossover)
    "macd_bearish",              # MACD < signal (no fresh crossover)
    # Volatility / bands
    "bollinger_above_upper",     # %B >= 1.0
    "bollinger_near_upper",      # 0.8 <= %B < 1.0
    "bollinger_middle",
    "bollinger_near_lower",      # 0.0 < %B <= 0.2
    "bollinger_below_lower",     # %B <= 0.0
    # Divergence
    "bullish_divergence",
    "bearish_divergence",
    # Position vs recent extremes
    "near_20d_high",             # within 2% of 20d high
    "near_20d_low",              # within 2% of 20d low
    "near_52w_high",
    "near_52w_low",
    # Trend regime
    "strong_uptrend",            # above MA-50 AND MA-200
    "strong_downtrend",          # below MA-50 AND MA-200
    "trend_mixed",               # one above, one below
    # Volatility regime
    "high_volatility",           # vol_regime = elevated
    "low_volatility",            # vol_regime = low
    "normal_volatility",
    # Momentum exhaustion
    "up_streak_3plus",
    "down_streak_3plus",
    # Sentiment context
    "already_priced_in",         # stock_vs_sector_1d strongly aligned with news direction
]


# -----------------------------------------------------------------------------
#  Internal formatting helpers
# -----------------------------------------------------------------------------

def _v(val, fmt: str = "+.2f", fallback: str = "N/A") -> str:
    if val is None:
        return fallback
    try:
        return format(val, fmt)
    except (TypeError, ValueError):
        return str(val)


def _trend_label(ps: dict) -> str:
    vs50  = ps.get("pct_vs_ma50")
    vs200 = ps.get("pct_vs_ma200")
    if vs50 is None or vs200 is None:
        return "unknown"
    if vs50 > 0 and vs200 > 0:  return "uptrend (above MA-50 and MA-200)"
    if vs50 < 0 and vs200 < 0:  return "downtrend (below MA-50 and MA-200)"
    if vs50 < 0 < vs200:        return "weakening (above MA-200 but below MA-50)"
    return "recovering (above MA-50 but below MA-200)"


def _vol_interpretation(ps: dict) -> str:
    regime = ps.get("vol_regime", "N/A")
    return {
        "elevated": "Elevated -> news impact AMPLIFIED.",
        "normal":   "Normal -> impact PROPORTIONAL to catalyst.",
        "low":      "Low -> impact MUTED unless structurally significant.",
    }.get(regime, "Unknown.")


def _format_context_block(ctx: dict) -> str:
    ps  = ctx["price_state"]
    ms  = ctx["market_state"]
    cs  = ctx["correlation_state"]
    ars = ctx.get("recent_abnormal_returns", [])

    peer_lines = "\n".join(
        f"  - {p['ticker']} | {p['sector']} | 90d-corr: {p['correlation']}"
        for p in cs["top_3_correlated_peers"]
    ) or "  (none)"

    ar_lines = "\n".join(
        f"  {r['date']}: stock {_v(r['stock_return_pct'])}%  "
        f"sector {_v(r['sector_return_pct'])}%  AR {_v(r['abnormal_return_pct'])}%"
        for r in ars
    ) or "  (insufficient data)"

    return (
        f"=== COMPANY ===\n"
        f"Ticker      : {ctx['ticker']}\n"
        f"Name        : {ctx['company']}\n"
        f"Sector      : {ctx['sector']}\n"
        f"Description : {ctx['business_description']}\n"
        f"Price data  : as of {ctx['last_price_date']}  (article date: {ctx['event_date']})\n"
        f"\n"
        f"=== PRICE STATE ===\n"
        f"Last close  : {_v(ps['last_close'], '.2f')} EGP\n"
        f"Return 1d   : {_v(ps['return_1d'])}%\n"
        f"Return 5d   : {_v(ps['return_5d'])}%\n"
        f"Return 20d  : {_v(ps['return_20d'])}%\n"
        f"Return 60d  : {_v(ps['return_60d'])}%\n"
        f"vs MA-50    : {_v(ps['pct_vs_ma50'])}%\n"
        f"vs MA-200   : {_v(ps['pct_vs_ma200'])}%\n"
        f"Trend       : {_trend_label(ps)}\n"
        f"Vol 20d ann : {_v(ps['vol_20d_annualized_pct'], '.2f')}%  [{ps.get('vol_regime', 'N/A')} regime]  {_vol_interpretation(ps)}\n"
        f"\n"
        f"=== SECTOR / MARKET ===\n"
        f"Sector ret 1d   : {_v(ms['sector_return_1d'])}%\n"
        f"Sector ret 5d   : {_v(ms['sector_return_5d'])}%\n"
        f"Stock vs sector 1d : {_v(ms['relative_strength_vs_sector_1d'])}%  (negative = underperforming)\n"
        f"\n"
        f"=== CORRELATION (90d) ===\n"
        f"With sector : {_v(cs['corr_with_sector_90d'], '.2f')}\n"
        f"Top peers:\n{peer_lines}\n"
        f"\n"
        f"=== RECENT ABNORMAL RETURNS (oldest -> newest) ===\n"
        f"{ar_lines}"
    )


def _format_technical_signals(ctx: dict) -> str:
    """Render the new technical_signals block in a compact, scannable form."""
    ts = ctx.get("technical_signals") or {}
    macd  = ts.get("macd")        or {}
    boll  = ts.get("bollinger")   or {}
    dist  = ts.get("distance_to_extremes") or {}

    macd_line  = f"  state={macd.get('state', 'N/A'):<24}  line={_v(macd.get('line'), '+.4f')}  signal={_v(macd.get('signal'), '+.4f')}  hist={_v(macd.get('histogram'), '+.4f')}"
    boll_line  = f"  zone={boll.get('zone', 'N/A'):<14}  %B={_v(boll.get('pct_b'), '.3f')}  bandwidth={_v(boll.get('bandwidth_pct'), '.2f')}%"
    dist_line  = (
        f"  20d high: {_v(dist.get('20d_high_distance_pct'))}%   20d low: {_v(dist.get('20d_low_distance_pct'))}%\n"
        f"  60d high: {_v(dist.get('60d_high_distance_pct'))}%   60d low: {_v(dist.get('60d_low_distance_pct'))}%\n"
        f"  52w high: {_v(dist.get('252d_high_distance_pct'))}%   52w low: {_v(dist.get('252d_low_distance_pct'))}%"
    )

    streak = ts.get("consecutive_streak", 0)
    streak_desc = (
        f"{'+' if streak >= 0 else ''}{streak}  "
        f"({'up' if streak > 0 else 'down' if streak < 0 else 'flat'} streak)"
    )

    return (
        f"=== TECHNICAL SIGNALS ===\n"
        f"RSI-14      : {_v(ts.get('rsi_14'), '.2f')}  [{ts.get('rsi_zone', 'N/A')}]\n"
        f"MACD (12/26/9):\n{macd_line}\n"
        f"Bollinger (20, 2sigma):\n{boll_line}\n"
        f"Distance to extremes (negative = below high, positive = above low):\n{dist_line}\n"
        f"RSI divergence : {ts.get('rsi_divergence', 'N/A')}\n"
        f"ATR (14d, avg |daily%|) : {_v(ts.get('atr_pct_14d'), '.2f')}%   <-- TYPICAL DAILY MOVE ANCHOR\n"
        f"Consecutive streak     : {streak_desc}"
    )


def _format_article(article: dict) -> str:
    return (
        f"Source    : {article.get('source', 'N/A')}\n"
        f"Published : {article.get('datetime') or article.get('date', 'N/A')}\n"
        f"Headline  : {article.get('title', '')}\n"
        f"\n"
        f"{article.get('body') or article.get('title', '')}"
    )


# -----------------------------------------------------------------------------
#  Reusable prompt blocks
# -----------------------------------------------------------------------------

def _ta_vocab_block() -> str:
    """Render the closed TA-signal vocabulary as a comma-separated list."""
    return "ta_signals_cited MUST be a subset of:\n  " + ", ".join(TA_SIGNAL_VOCAB)


_HORIZON_BLOCK = """\
=== HORIZONS (predict BOTH) ===
short  -> t+1 to t+2  trading days  (neutral band +-1.5%; mag: <1.5% / 1.5-4% / >=4%)
medium -> t+1 to t+20 trading days  (neutral band +-4%;   mag: <4%   / 4-8%   / >=8%)

Magnitude is |cumulative abnormal return| = sum of (stock - sector) daily
returns over the window, NOT raw price change. Direction is relative to the
sector benchmark too -- "up" means OUTPERFORMING the sector.
"""


_TA_CHECKLIST = """\
=== TECHNICAL CHECKLIST (walk through ALL FIVE before predicting) ===

T1. OVERBOUGHT / OVERSOLD
    rsi_overbought OR bollinger_above_upper OR near_20d_high
       -> bullish news likely capped; bearish news amplified.
    rsi_oversold OR bollinger_below_lower OR near_20d_low
       -> bearish news likely capped; bullish news amplified (bounce).

T2. DIVERGENCE (mean-reversion warning)
    bearish_divergence + bullish news  -> react LESS than expected (prefer small/neutral).
    bullish_divergence + bearish news  -> react LESS than expected (prefer small/neutral).
    Divergence is a STRONG counter-signal -- do not ignore.

T3. MOMENTUM CONFIRMATION (MACD)
    macd_bullish_crossover + bullish news -> amplify direction & magnitude.
    macd_bearish_crossover + bearish news -> amplify direction & magnitude.
    MACD state contradicts the news -> lower confidence, narrow magnitude.

T4. ROOM TO RUN (distance_to_extremes)
    At/near 20d high -> upside structurally capped (resistance).
    At/near 20d low  -> downside structurally capped (support).
    Cite the exact % distance to support your magnitude call.

T5. MAGNITUDE ANCHOR (most important -- this fixes the "always-small" failure mode)
    atr_pct_14d = THIS stock's typical daily abnormal-return scale.
    Over 2 days  the noise floor is ~1.4 x atr_pct_14d.
    Over 20 days the noise floor is ~4.5 x atr_pct_14d.
    Example: atr_pct_14d = 2.0% -> 2-day noise ~2.8%, 20-day noise ~9%.
    A "small" prediction means you expect the news to SUPPRESS normal vol.
    A "large" prediction means signal clearly EXCEEDS the noise floor.
    Predicting "small" without referencing atr_pct_14d is a calibration error.
"""


_REASONING_RULES = """\
=== REASONING RULES ===
R1. CITE WITH LOGIC. Every TA signal in ta_signals_cited must be relevant to
    YOUR prediction -- not filler.
R2. NEWS-TYPE CHECKLIST (quick).
    capital_increase / m_and_a -> quantify dilution = new_shares / pre_existing
                                  (state BOTH inputs). >5% dilution without
                                  preemptive rights = bearish prior.
    earnings   -> compare to consensus / recent trend.
    dividend   -> quantify yield = div / last_close.
    regulatory -> approval positive; probe/fine negative.
    operational-> contract/win positive; halt/accident negative.
    macro      -> lean on sector sensitivity.
R3. NO HALLUCINATION. State both inputs for any ratio.
R4. PRICED-IN. If stock_vs_sector_1d is strongly aligned with the news
    direction, SHORT magnitude defaults to "small" (mean-reversion possible).
R5. CONFIDENCE: 0.8-1.0 clear+aligned; 0.5-0.7 mixed; 0.3-0.5 ambiguous;
    0.0-0.3 undetermined. Default conf_medium <= conf_short.
R6. NO LANGUAGE LEAKS. Arabic free-text fields are 100% Arabic.
"""


# -----------------------------------------------------------------------------
#  1.  IMPACT AGENT PROMPT
# -----------------------------------------------------------------------------

def impact_agent_prompt(
    ctx: dict,
    sentiment_score: float,
    sentiment_label: str,
    article: dict,
) -> str:
    return f"""\
You are a quantitative analyst predicting Egyptian Stock Exchange (EGX) stock \
reactions to Arabic news. Your prediction must be grounded in the technical \
signals provided -- not vibes.

{_format_context_block(ctx)}

{_format_technical_signals(ctx)}

=== ARABIC SENTIMENT (one input among many) ===
Label : {sentiment_label}
Score : {_v(sentiment_score, '+.4f')}   (-1 strongly negative -> +1 strongly positive)
NOTE: CAMeLBERT classifies tone, not financial implication.

=== NEWS ARTICLE (Arabic -- do NOT translate) ===
{_format_article(article)}

{_HORIZON_BLOCK}

{_TA_CHECKLIST}

{_REASONING_RULES}

{_ta_vocab_block()}

=== OUTPUT INSTRUCTIONS ===
1. Classify news_type FIRST.
2. Walk through T1-T5 in order. Pick the TA signals that ACTUALLY apply from
   the closed vocab list above (no inventions).
3. Anchor magnitudes to atr_pct_14d (T5). State the anchor in overall_reasoning.
4. Quantify any dilution/yield ratio with both inputs.
5. Confidence: default conf_medium <= conf_short.
6. LANGUAGE: All free-text Arabic. JSON keys and enum values in English.
7. Output ONLY the JSON below. No markdown fences. No text outside.

{{
  "news_type":         "capital_increase" | "m_and_a" | "earnings" | "dividend" | "regulatory" | "operational" | "macro" | "other",
  "already_priced_in": <true | false>,
  "quantified_ratio":  "<Arabic ratio with both inputs OR 'لا ينطبق'>",
  "key_drivers":       ["<driver 1 Arabic>", "<driver 2 Arabic>"],
  "ta_signals_cited":  ["<signal from closed vocab>", "..."],
  "per_horizon": {{
    "short": {{
      "direction":  "up" | "down" | "neutral",
      "magnitude":  "small" | "medium" | "large",
      "confidence": <float 0.0-1.0>,
      "reasoning":  "<1-2 Arabic sentences. Cite >=1 TA signal by name.>"
    }},
    "medium": {{
      "direction":  "up" | "down" | "neutral",
      "magnitude":  "small" | "medium" | "large",
      "confidence": <float 0.0-1.0>,
      "reasoning":  "<1-2 Arabic sentences. Cite >=1 TA signal by name.>"
    }}
  }},
  "overall_reasoning": "<3-4 Arabic sentences. Walk T1-T5 briefly. Reference atr_pct_14d as the magnitude anchor. State the quantified ratio if applicable.>"
}}"""


# -----------------------------------------------------------------------------
#  2.  CRITIC AGENT PROMPT  (Claude)
# -----------------------------------------------------------------------------

def critic_agent_prompt(
    ctx: dict,
    sentiment_score: float,
    sentiment_label: str,
    article: dict,
    impact_output: dict,
) -> str:
    impact_json = json.dumps(impact_output, ensure_ascii=False, indent=2)

    return f"""\
You are a senior quantitative risk analyst at an Egyptian investment bank. \
A junior LLM analyst has assessed a news event about {ctx['ticker']} ({ctx['company']}). \
Your job:
  (a) audit the junior's TA citations against the live data,
  (b) check magnitude vs the ATR anchor and priced-in vs the 1d move,
  (c) produce the final authoritative prediction across short and medium horizons,
  (d) write a plain-Arabic explanation for a non-expert retail investor.

{_format_context_block(ctx)}

{_format_technical_signals(ctx)}

=== ARABIC SENTIMENT ===
Label : {sentiment_label}
Score : {_v(sentiment_score, '+.4f')}

=== NEWS ARTICLE (Arabic -- do NOT translate) ===
{_format_article(article)}

=== IMPACT AGENT OUTPUT (junior, Groq) ===
{impact_json}

{_HORIZON_BLOCK}

{_TA_CHECKLIST}

{_REASONING_RULES}

{_ta_vocab_block()}

=== REVIEW CHECKLIST ===
C1. NEWS_TYPE: correct?
C2. TA_SIGNALS: every cited signal must be supported by the data block above.
    If junior cited "rsi_overbought" but RSI is 45, REMOVE the signal and note
    the override.
C3. MAGNITUDE vs ATR: does the predicted magnitude make sense relative to
    1.4 x atr_pct_14d (short) and 4.5 x atr_pct_14d (medium)? Adjust if mis-scaled.
C4. PRICED-IN: if stock_vs_sector_1d aligns with the news direction, short
    magnitude should default to "small" (R4).
C5. DIVERGENCE: if a divergence signal is present, did the junior account for it?
C6. CONFIDENCE: lower it when agents-disagree-with-context or when news is priced in.
    Default conf_medium <= conf_short.
C7. LANGUAGE LEAKS: strip any non-Arabic word from Arabic fields (R6).

=== USER-FACING EXPLANATION (mandatory) ===
After the technical analysis, produce a plain-Arabic explanation for a
non-expert retail investor.

U1. NO JARGON. Replace technical terms with everyday Arabic:
    "تخفيف" -> "تقليل نسبة ملكية المساهمين الحاليين"
    "MA-50/MA-200" -> "متوسط سعر السهم خلال الشهرين / السنة الماضية"
    "RSI" -> "مؤشر قوة الشراء والبيع"
    "MACD" -> "مؤشر تقاطع المتوسطات"
    "Bollinger" -> "نطاقات تذبذب السعر"
    "ATR" -> "متوسط الحركة اليومية المعتادة"
U2. NUMBERS WITH MEANING. Every cited number gets a plain-Arabic interpretation.
U3. STATE THE PREDICTION FIRST inside what_we_expect (direction + size + timeframe),
    THEN 1-2 sentences of supporting evidence.
U4. PER-HORIZON. outlook_by_horizon.short and .medium must each be ONE Arabic
    sentence with direction + size + rationale clue.
U5. BREVITY. ~120-180 Arabic words across the whole user block.
U6. NO LANGUAGE LEAKS. 100% Arabic in the user block.

=== INSTRUCTIONS ===
1. Override the junior anywhere evidence demands it; list each change in
   "disagreements_with_junior" with the reason.
2. Confidence: adjust per horizon. Default conf_medium <= conf_short.
3. SPELLING: "RSI-14" not "RS-14". Proper Arabic punctuation.
4. LANGUAGE: All free-text Arabic; JSON keys + tickers + enum values in English.
5. Output ONLY the JSON below. No markdown fences. No text outside.

{{
  "primary_company": {{
    "ticker":              "{ctx['ticker']}",
    "news_type":           "capital_increase" | "m_and_a" | "earnings" | "dividend" | "regulatory" | "operational" | "macro" | "other",
    "already_priced_in":   <true | false>,
    "quantified_ratio":    "<Arabic ratio with both inputs OR 'لا ينطبق'>",
    "ta_signals_cited":    ["<signal from closed vocab>", "..."],
    "per_horizon": {{
      "short": {{
        "direction":  "up" | "down" | "neutral",
        "magnitude":  "small" | "medium" | "large",
        "confidence": <float 0.0-1.0>,
        "reasoning":  "<1-2 Arabic sentences citing TA signals by name>"
      }},
      "medium": {{
        "direction":  "up" | "down" | "neutral",
        "magnitude":  "small" | "medium" | "large",
        "confidence": <float 0.0-1.0>,
        "reasoning":  "<1-2 Arabic sentences citing TA signals by name>"
      }}
    }},
    "overall_reasoning":   "<3-4 Arabic sentences. Reference atr_pct_14d anchor. State news_type. Note any override of the junior.>"
  }},
  "risk_flags": [
    "<event-specific risk in Arabic>",
    "<event-specific risk in Arabic>"
  ],
  "disagreements_with_junior": [
    "<what the junior said, what is correct, why -- in Arabic>"
  ],
  "external_signal_alignment":   "aligned" | "partially_aligned" | "contradicts",
  "external_signal_explanation": "<1-2 sentences in Arabic>",
  "tldr": {{
    "verdict":   "<short Arabic phrase: direction + horizon>",
    "one_line":  "<one Arabic sentence summarising the whole view>"
  }},
  "explanation_for_user": {{
    "headline":                  "<one sentence plain Arabic, no jargon>",
    "what_happened":             "<2-3 sentences plain Arabic, no jargon>",
    "why_it_matters":            "<2-3 sentences plain Arabic, financial logic in everyday terms>",
    "what_we_expect":            "<lead with prediction (direction + size + timeline) THEN evidence>",
    "outlook_by_horizon": {{
      "short":  "<one Arabic sentence: direction + size + rationale for the next 2 days>",
      "medium": "<one Arabic sentence: direction + size + rationale for the next month>"
    }},
    "what_could_change_our_view":"<1-2 Arabic sentences on key uncertainties>"
  }}
}}"""


# -----------------------------------------------------------------------------
#  Smoke test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from features import compute_context

    ctx = compute_context("EMFD", "2025-01-02")
    dummy_article = {
        "title": "إعمار مصر تُقر زيادة رأس المال للاستحواذ على البرو نورث كوست",
        "body":  "آراب فاينانس: وافق مجلس إدارة شركة إعمار مصر...",
        "source": "Arab Finance",
        "date":   "2025-01-02",
    }
    print(impact_agent_prompt(ctx, 0.0579, "neutral", dummy_article))
