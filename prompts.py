# prompts.py
"""
Prompt templates for the two-agent EGX30 news-impact pipeline (v2.1).

    impact_agent_prompt() -> str   for Groq (Qwen3-32B or similar)
    critic_agent_prompt() -> str   for Claude (Anthropic)

HORIZONS (three buckets, all from the same START day):
    short  -> 1  trading day  from start
    medium -> 5  trading days from start
    large  -> 20 trading days from start

START DAY rule (EGX hours 10:00 - 14:30 Cairo):
    If the news arrived DURING trading hours -> start = event_date itself.
    Otherwise                                  -> start = the next trading day.

Magnitude bands are PER-COMPANY, scaled by that ticker's ATR-14d. A high-vol
ticker (BTFH) gets wider bands than a low-vol staple (EFIH). Bands are
computed once in features.compute_context and stored in ctx["dynamic_bands"];
both the prompt and the evaluator read them from there.

Dividend handling: when news_type=dividend, the impact agent extracts
`dividend_info.yield_pct` and `dividend_info.ex_div_date` from the article so
the evaluator can neutralise the mechanical ex-div price drop.
"""

import json

# Horizons + Arabic labels are sourced from features.py -- single edit point.
from features import HORIZON_TRADING_DAYS, HORIZON_LABEL_AR

HORIZONS = list(HORIZON_TRADING_DAYS.keys())


# -----------------------------------------------------------------------------
#  Closed TA-signal vocabulary  (the model MUST pick from this list)
# -----------------------------------------------------------------------------

TA_SIGNAL_VOCAB = [
    # Momentum / oscillators
    "rsi_overbought",            # rsi_14 > 70
    "rsi_oversold",              # rsi_14 < 30
    "rsi_neutral",
    "macd_bullish_crossover",
    "macd_bearish_crossover",
    "macd_bullish",
    "macd_bearish",
    # Volatility / bands
    "bollinger_above_upper",
    "bollinger_near_upper",
    "bollinger_middle",
    "bollinger_near_lower",
    "bollinger_below_lower",
    # Divergence
    "bullish_divergence",
    "bearish_divergence",
    # Position vs recent extremes
    "near_20d_high",
    "near_20d_low",
    "near_52w_high",
    "near_52w_low",
    # Trend regime
    "strong_uptrend",
    "strong_downtrend",
    "trend_mixed",
    # Volatility regime
    "high_volatility",
    "low_volatility",
    "normal_volatility",
    # Momentum exhaustion
    "up_streak_3plus",
    "down_streak_3plus",
    # Sentiment context
    "already_priced_in",
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
    if vs50 is None or vs200 is None:  return "unknown"
    if vs50 > 0 and vs200 > 0:         return "uptrend (above MA-50 and MA-200)"
    if vs50 < 0 and vs200 < 0:         return "downtrend (below MA-50 and MA-200)"
    if vs50 < 0 < vs200:               return "weakening (above MA-200 but below MA-50)"
    return "recovering (above MA-50 but below MA-200)"


def _vol_interpretation(ps: dict) -> str:
    return {
        "elevated": "Elevated -> impact AMPLIFIED.",
        "normal":   "Normal -> impact PROPORTIONAL to catalyst.",
        "low":      "Low -> impact MUTED unless structurally significant.",
    }.get(ps.get("vol_regime", "N/A"), "Unknown.")


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
        f"Return 1d   : {_v(ps['return_1d'])}%   5d : {_v(ps['return_5d'])}%   "
        f"20d : {_v(ps['return_20d'])}%   60d : {_v(ps['return_60d'])}%\n"
        f"vs MA-50    : {_v(ps['pct_vs_ma50'])}%  vs MA-200 : {_v(ps['pct_vs_ma200'])}%\n"
        f"Trend       : {_trend_label(ps)}\n"
        f"Vol 20d ann : {_v(ps['vol_20d_annualized_pct'], '.2f')}%  [{ps.get('vol_regime', 'N/A')} regime]  {_vol_interpretation(ps)}\n"
        f"\n"
        f"=== SECTOR ===\n"
        f"Sector 1d   : {_v(ms['sector_return_1d'])}%  5d : {_v(ms['sector_return_5d'])}%\n"
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
    ts   = ctx.get("technical_signals") or {}
    macd = ts.get("macd") or {}
    boll = ts.get("bollinger") or {}
    dist = ts.get("distance_to_extremes") or {}

    streak = ts.get("consecutive_streak", 0)
    streak_desc = (
        f"{'+' if streak >= 0 else ''}{streak} "
        f"({'up' if streak > 0 else 'down' if streak < 0 else 'flat'} streak)"
    )

    return (
        f"=== TECHNICAL SIGNALS ===\n"
        f"RSI-14    : {_v(ts.get('rsi_14'), '.2f')}  [{ts.get('rsi_zone', 'N/A')}]\n"
        f"MACD      : state={macd.get('state', 'N/A')}  hist={_v(macd.get('histogram'), '+.4f')}\n"
        f"Bollinger : zone={boll.get('zone', 'N/A')}  %B={_v(boll.get('pct_b'), '.3f')}\n"
        f"Distance  : 20d_hi {_v(dist.get('20d_high_distance_pct'))}%  20d_lo {_v(dist.get('20d_low_distance_pct'))}%  "
        f"60d_hi {_v(dist.get('60d_high_distance_pct'))}%  60d_lo {_v(dist.get('60d_low_distance_pct'))}%  "
        f"52w_hi {_v(dist.get('252d_high_distance_pct'))}%  52w_lo {_v(dist.get('252d_low_distance_pct'))}%\n"
        f"Divergence : {ts.get('rsi_divergence', 'N/A')}\n"
        f"ATR 14d   : {_v(ts.get('atr_pct_14d'), '.2f')}%  <-- TYPICAL DAILY MOVE FOR THIS TICKER\n"
        f"Streak    : {streak_desc}"
    )


def _format_dynamic_bands(ctx: dict) -> str:
    """Render the per-company magnitude / neutral bands as a compact reference."""
    db = ctx.get("dynamic_bands") or {}
    nb = db.get("neutral_band_pct") or {}
    mb = db.get("magnitude_bands_pct") or {}
    atr = db.get("atr_pct_anchor")

    rows = []
    for h in HORIZONS:
        bands = mb.get(h) or {}
        sm    = bands.get("small")   or [0, 0]
        med   = bands.get("medium")  or [0, 0]
        lg    = bands.get("large")   or [0, None]
        rows.append(
            f"  {h:<7} (N={HORIZON_TRADING_DAYS[h]:>2}d)  "
            f"neutral=+-{_v(nb.get(h), '.2f')}%  "
            f"small <{_v(sm[1], '.2f')}%  "
            f"medium {_v(med[0], '.2f')}%-{_v(med[1], '.2f')}%  "
            f"large >={_v(lg[0], '.2f')}%"
        )
    return (
        f"=== PER-COMPANY MAGNITUDE BANDS (anchored to this ticker's ATR={_v(atr, '.2f')}%) ===\n"
        + "\n".join(rows)
        + "\n(These bands ARE THE TRUTH for this prediction. A 3% move is 'large' "
          "for a low-vol ticker and 'medium' for a high-vol one. Use them.)"
    )


def _format_news_timing(ctx: dict) -> str:
    """Tell the model when the news arrived (controls window start day)."""
    timing = ctx.get("news_timing") or {}
    in_hours    = timing.get("in_trading_hours")
    arrived_at  = timing.get("arrived_at_str", "unknown")
    start_day   = timing.get("start_day_label", "unknown")
    horizon_summary = " ; ".join(
        f"{h}={HORIZON_TRADING_DAYS[h]}d" for h in HORIZONS
    )
    return (
        f"=== NEWS TIMING ===\n"
        f"Arrived at      : {arrived_at}\n"
        f"In trading hours: {in_hours}  (EGX hours: 10:00 - 14:30 Cairo)\n"
        f"Window start    : {start_day}\n"
        f"Horizons (days) : {horizon_summary}"
    )


def _format_article(article: dict) -> str:
    return (
        f"Source    : {article.get('source', 'N/A')}\n"
        f"Published : {article.get('datetime') or article.get('date', 'N/A')}\n"
        f"Headline  : {article.get('title', '')}\n"
        f"\n"
        f"{article.get('body') or article.get('title', '')}"
    )


def _ta_vocab_block() -> str:
    return "ta_signals_cited MUST be a subset of:\n  " + ", ".join(TA_SIGNAL_VOCAB)


# -----------------------------------------------------------------------------
#  Shared blocks
# -----------------------------------------------------------------------------

def _horizon_block() -> str:
    """
    Build the HORIZONS section dynamically from HORIZON_TRADING_DAYS so
    editing horizons in features.py needs no prompt rewrite.
    """
    lines = []
    for h in HORIZONS:
        n = HORIZON_TRADING_DAYS[h]
        if n == 1:
            window = "t+0 only (just the start day's return)"
        else:
            window = f"t+0 .. t+{n-1}  ({n} trading days cumulative)"
        label = HORIZON_LABEL_AR.get(h, "")
        lines.append(f"  {h:<7} -> {window:<40}  AR label: {label}")
    return (
        "=== HORIZONS (predict ALL OF THEM) ===\n"
        + "\n".join(lines)
        + "\n\n"
          "Magnitude = |cumulative abnormal return| = sum of (stock - sector) daily\n"
          "returns over the window. Direction is RELATIVE TO THE SECTOR -- 'up'\n"
          "means OUTPERFORMING the sector benchmark.\n"
    )


_TA_CHECKLIST = """\
=== TECHNICAL CHECKLIST (walk through ALL FIVE before predicting) ===

T1. OVERBOUGHT / OVERSOLD
    rsi_overbought OR bollinger_above_upper OR near_20d_high
       -> bullish news likely capped; bearish news amplified.
    rsi_oversold OR bollinger_below_lower OR near_20d_low
       -> bearish news likely capped; bullish news amplified (bounce).

T2. DIVERGENCE (mean-reversion warning)
    bearish_divergence + bullish news  -> react LESS than expected.
    bullish_divergence + bearish news  -> react LESS than expected.

T3. MOMENTUM CONFIRMATION (MACD)
    macd_bullish_crossover + bullish news -> amplify direction & magnitude.
    macd_bearish_crossover + bearish news -> amplify direction & magnitude.
    MACD state contradicts the news -> lower confidence.

T4. ROOM TO RUN (distance_to_extremes)
    At/near 20d high -> upside structurally capped.
    At/near 20d low  -> downside structurally capped.
    Cite the % distance to support your magnitude call.

T5. PER-COMPANY MAGNITUDE BANDS (most important calibration fix)
    The "small/medium/large" bands above are PER-COMPANY, scaled by atr_pct_14d.
    A high-vol ticker (BTFH, ATR ~ 4%) has 'large' = >8% at short; a low-vol
    staple (EFIH, ATR ~ 1%) has 'large' = >2% at short. USE THE TICKER'S OWN
    BANDS -- not a global rule of thumb.

    Sanity anchor: typical noise floor over N days = ATR x sqrt(N).
       short  (1d):  ~1.0 x ATR noise.
       medium (5d):  ~2.2 x ATR noise.
       large  (20d): ~4.5 x ATR noise.
    Predicting 'large' means the news signal clearly EXCEEDS the noise floor.
"""


_REASONING_RULES = """\
=== REASONING RULES ===
R1. CITE WITH LOGIC. Every TA signal in ta_signals_cited must be relevant.
R2. NEWS-TYPE CHECKLIST (quick).
    capital_increase / m_and_a -> quantify dilution = new_shares / pre_existing
                                  (state BOTH inputs). >5% no preemptive = bearish.
    earnings   -> compare to consensus or recent trend.
    dividend   -> quantify yield = div_per_share / last_close; extract ex-div date.
    regulatory -> approval positive; probe/fine negative.
    operational-> contract/win positive; halt/accident negative.
    macro      -> lean on sector sensitivity.
R3. NO HALLUCINATION. State both inputs for any ratio.
R4. PRICED-IN. If stock_vs_sector_1d strongly aligns with news direction,
    SHORT magnitude defaults to "small" (mean-reversion possible).
R5. CONFIDENCE: 0.8-1.0 clear; 0.5-0.7 mixed; 0.3-0.5 ambiguous; 0.0-0.3 undetermined.
    Default conf_large <= conf_medium <= conf_short.
R6. NO LANGUAGE LEAKS. Arabic free-text fields are 100% Arabic.
R7. DIVIDEND EX-DATE. If news_type=dividend, FILL dividend_info with the
    yield % and (if stated) the ex-div date. Note: the mechanical price drop
    on ex-div day is NOT a stock reaction -- the evaluator will neutralise it.
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
signals provided and calibrated to this specific ticker's volatility.

{_format_context_block(ctx)}

{_format_technical_signals(ctx)}

{_format_dynamic_bands(ctx)}

{_format_news_timing(ctx)}

=== ARABIC SENTIMENT (one input among many) ===
Label : {sentiment_label}
Score : {_v(sentiment_score, '+.4f')}   (-1 strongly negative -> +1 strongly positive)
NOTE: CAMeLBERT classifies tone, not financial implication.

=== NEWS ARTICLE (Arabic -- do NOT translate) ===
{_format_article(article)}

{_horizon_block()}

{_TA_CHECKLIST}

{_REASONING_RULES}

{_ta_vocab_block()}

=== OUTPUT INSTRUCTIONS ===
1. Classify news_type FIRST.
2. Walk through T1-T5. Pick TA signals that ACTUALLY apply from the closed vocab.
3. Anchor magnitudes to the PER-COMPANY bands above (NOT a global rule).
4. Quantify any dilution/yield ratio with both inputs.
5. If news_type=dividend, FILL dividend_info; otherwise set it to null.
6. LANGUAGE: free-text Arabic. JSON keys + enum values in English.
7. Output ONLY the JSON below. No markdown fences. No text outside.

{{
  "news_type":         "capital_increase" | "m_and_a" | "earnings" | "dividend" | "regulatory" | "operational" | "macro" | "other",
  "already_priced_in": <true | false>,
  "quantified_ratio":  "<Arabic ratio with both inputs OR 'لا ينطبق'>",
  "key_drivers":       ["<driver 1 Arabic>", "<driver 2 Arabic>"],
  "ta_signals_cited":  ["<signal from closed vocab>", "..."],
  "dividend_info": null | {{
    "yield_pct":      <float>,
    "ex_div_date":    "<YYYY-MM-DD or null if unknown>",
    "amount_egp":     <float or null>
  }},
  "per_horizon": {{
    "short":  {{"direction":"up|down|neutral", "magnitude":"small|medium|large", "confidence":<0-1>, "reasoning":"<1-2 Arabic sentences citing >=1 TA signal>"}},
    "medium": {{"direction":"up|down|neutral", "magnitude":"small|medium|large", "confidence":<0-1>, "reasoning":"<1-2 Arabic sentences citing >=1 TA signal>"}},
    "large":  {{"direction":"up|down|neutral", "magnitude":"small|medium|large", "confidence":<0-1>, "reasoning":"<1-2 Arabic sentences citing >=1 TA signal>"}}
  }},
  "overall_reasoning": "<3-4 Arabic sentences. Walk T1-T5 briefly. Reference the per-company magnitude bands explicitly. State the quantified ratio if applicable.>"
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
  (b) check magnitudes against the PER-COMPANY bands (not global rules),
  (c) produce the final authoritative prediction across short / medium / large,
  (d) write a plain-Arabic explanation for a non-expert retail investor.

{_format_context_block(ctx)}

{_format_technical_signals(ctx)}

{_format_dynamic_bands(ctx)}

{_format_news_timing(ctx)}

=== ARABIC SENTIMENT ===
Label : {sentiment_label}
Score : {_v(sentiment_score, '+.4f')}

=== NEWS ARTICLE (Arabic -- do NOT translate) ===
{_format_article(article)}

=== IMPACT AGENT OUTPUT (junior, Groq) ===
{impact_json}

{_horizon_block()}

{_TA_CHECKLIST}

{_REASONING_RULES}

{_ta_vocab_block()}

=== REVIEW CHECKLIST ===
C1. NEWS_TYPE correct?
C2. TA_SIGNALS: every cited signal must be supported by the data. Override
    inconsistent cites silently in your final output.
C3. MAGNITUDE vs PER-COMPANY BANDS: does the predicted bucket match the
    horizon's per-company thresholds? Adjust if mis-scaled.
C4. PRICED-IN: if stock_vs_sector_1d aligns with the news direction, short
    magnitude should default to "small" (R4).
C5. DIVERGENCE: if a divergence signal is present, did the junior account for it?
C6. CONFIDENCE: lower when context contradicts the call or news is priced in.
    Default conf_large <= conf_medium <= conf_short.
C7. DIVIDEND_INFO: if news_type=dividend, ensure yield_pct is present. If the
    article gives the ex-div date, extract it as YYYY-MM-DD; otherwise null.
C8. LANGUAGE: strip any non-Arabic word from Arabic fields (R6).

=== USER-FACING EXPLANATION (the most important deliverable) ===
You will produce TWO reasoning blocks tuned to two audiences:

  news_story:       PLAIN ARABIC, ZERO JARGON. Just the news content and what
                    it means in everyday language. A retail investor with no
                    finance training must understand it. No RSI / MACD / ATR.

  technical_view:   Includes TA signals AND explains each term in Arabic
                    inline. Format example:
                      "مؤشر RSI (مقياس قوة الشراء والبيع) عند 72 وهو مرتفع..."
                      "مؤشر MACD (تقاطع المتوسطات المتحركة) في حالة هبوطية..."
                      "نطاقات بولينجر (مستوى التذبذب) تشير إلى تشبع..."
                      "مؤشر ATR (متوسط الحركة اليومية المعتادة) لهذا السهم 1.5%..."

  sentiment_note:   ONE Arabic sentence that surfaces the CAMeLBERT sentiment
                    result (label + score given above in 'ARABIC SENTIMENT')
                    AND explicitly says whether it agrees with our prediction
                    or contradicts it.

  outlook_by_horizon: ONE Arabic sentence per horizon. Direction + size + a
                    one-clause reason. Numbers stated in plain language.

  what_could_change_our_view: 1-2 Arabic sentences on key uncertainties.

U-rules:
  U1. NUMBERS WITH MEANING. Every cited number gets a plain-Arabic interpretation.
  U2. STATE THE PREDICTION FIRST in news_story (direction + timeline), THEN evidence.
  U3. BREVITY. Each field is short -- prefer 2-3 sentences max per field.
  U4. 100% Arabic in this entire block. The next agent (QA) will reject leaks.

=== INSTRUCTIONS ===
1. Override the junior wherever evidence demands; do not surface the override
   to the user -- just produce the corrected final prediction.
2. Confidence: adjust per horizon. Default conf_large <= conf_medium <= conf_short.
3. SPELLING: "RSI-14" not "RS-14". Proper Arabic punctuation.
4. LANGUAGE: free-text Arabic; JSON keys + tickers + enum values in English.
5. Output ONLY the JSON below. No markdown fences. No text outside.

{{
  "primary_company": {{
    "ticker":              "{ctx['ticker']}",
    "news_type":           "capital_increase" | "m_and_a" | "earnings" | "dividend" | "regulatory" | "operational" | "macro" | "other",
    "already_priced_in":   <true | false>,
    "quantified_ratio":    "<Arabic ratio with both inputs OR 'لا ينطبق'>",
    "ta_signals_cited":    ["<signal from closed vocab>", "..."],
    "dividend_info": null | {{
      "yield_pct":   <float>,
      "ex_div_date": "<YYYY-MM-DD or null>",
      "amount_egp":  <float or null>
    }},
    "per_horizon": {{
      "short":  {{"direction":"up|down|neutral", "magnitude":"small|medium|large", "confidence":<0-1>, "reasoning":"<1-2 Arabic sentences citing TA>"}},
      "medium": {{"direction":"up|down|neutral", "magnitude":"small|medium|large", "confidence":<0-1>, "reasoning":"<1-2 Arabic sentences citing TA>"}},
      "large":  {{"direction":"up|down|neutral", "magnitude":"small|medium|large", "confidence":<0-1>, "reasoning":"<1-2 Arabic sentences citing TA>"}}
    }},
    "overall_reasoning":   "<2-3 Arabic sentences. Reference the per-company bands. State news_type. Concise.>"
  }},
  "tldr": "<one Arabic sentence summarising the whole view: direction + dominant horizon>",
  "explanation_for_user": {{
    "headline":                   "<one sentence in plain Arabic, no jargon, captures the news>",
    "news_story":                 "<2-3 sentences PLAIN Arabic, zero technical jargon. What happened and why it matters to an everyday investor.>",
    "technical_view":             "<2-3 sentences with TA signals, EACH technical term explained in Arabic inline as shown above (RSI / MACD / Bollinger / ATR / divergence).>",
    "sentiment_note":             "<1 Arabic sentence: state the CAMeLBERT label and score, and whether it agrees with our prediction.>",
    "outlook_by_horizon": {{
      "short":  "<one Arabic sentence: direction + size + reason for t+0>",
      "medium": "<one Arabic sentence: direction + size + reason for t+0..t+1>",
      "large":  "<one Arabic sentence: direction + size + reason for t+0..t+19>"
    }},
    "what_could_change_our_view": "<1-2 Arabic sentences on key uncertainties>"
  }}
}}"""


# -----------------------------------------------------------------------------
#  3.  ARABIC-QA AGENT PROMPT  (final polish pass -- fast, Groq Llama)
# -----------------------------------------------------------------------------

def arabic_qa_prompt(explanation: dict) -> str:
    """
    Compact prompt for a professional Arabic language editor.
    Receives the critic's `explanation_for_user` dict (Arabic free-text fields)
    and returns the same dict with:
      - any non-Arabic characters (Chinese / Korean / Latin / German etc.) replaced
      - grammar + spelling corrected in Modern Standard Arabic
      - professional financial Arabic tone
      - structure / keys / numbers / ticker symbols PRESERVED

    Intentionally small so a fast Groq model (Llama 3.3 70B) returns in ~500ms.
    """
    raw = json.dumps(explanation, ensure_ascii=False, indent=2)
    return f"""\
You are a professional Arabic language editor (محرر لغوي محترف) specialising
in Egyptian financial Arabic. The JSON below contains analyst commentary that
will be shown to retail investors on an EGX30 stock service.

YOUR JOB:
1. SCAN every string value for non-Arabic content -- Chinese / Korean / Japanese
   characters, English words mid-sentence, German, French, transliterations
   ("Richtung", "spillover", "trend"), Latin punctuation that breaks Arabic flow.
2. REPLACE those with proper Modern Standard Arabic equivalents.
3. CORRECT grammar, spelling, and missing diacritics where helpful.
4. POLISH clumsy phrasing into clean, professional financial Arabic.
5. PRESERVE: JSON structure, all keys, all numeric values, ticker symbols
   (uppercase like EMFD, COMI, BTFH), the values of enum fields if any.
6. PRESERVE the original meaning -- you are an editor, not a re-writer.
   Do NOT change predictions or invent new facts.
7. ALL output in Arabic. Do NOT translate Arabic into English or vice-versa.

INPUT JSON:
{raw}

OUTPUT (corrected JSON only, no markdown fences, no commentary):"""


# -----------------------------------------------------------------------------
#  Smoke test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from features import compute_context, news_in_trading_hours

    ctx = compute_context("EMFD", "2025-01-02")
    art = {
        "title": "إعمار مصر تُقر زيادة رأس المال للاستحواذ على البرو نورث كوست",
        "body":  "آراب فاينانس: وافق مجلس إدارة شركة إعمار مصر...",
        "source": "Arab Finance",
        "date":   "2025-01-02",
        "time":   "10:54:00",
    }
    ctx["news_timing"] = {
        "in_trading_hours": news_in_trading_hours(art),
        "arrived_at_str":   art.get("datetime") or f"{art['date']} {art.get('time', '?')}",
        "start_day_label":  "event_date (article day, news during trading hours)",
    }
    print(impact_agent_prompt(ctx, 0.0579, "neutral", art))
