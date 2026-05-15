# prompts.py
"""
Prompt templates for the three-agent EGX30 news-impact pipeline.

    impact_agent_prompt()    -> str   for Groq (Qwen3-32B or similar)
    spillover_agent_prompt() -> str   for Groq (Qwen3-32B or similar)
    critic_agent_prompt()    -> str   for Claude (Anthropic)

Horizon scheme (locked):
    1d     -> evaluation at t+1   close, neutral band +-1%
    2-5d   -> evaluation at t+5   close, neutral band +-2%
    1-2w   -> evaluation at t+10  close, neutral band +-3%
    2-4w   -> evaluation at t+20  close, neutral band +-4%
    1-3m   -> evaluation at t+60  close, neutral band +-5%
"""

import json
from datetime import date, datetime, timedelta


# -----------------------------------------------------------------------------
#  Horizon configuration (single source of truth)
# -----------------------------------------------------------------------------

HORIZONS = ["1d", "2-5d", "1-2w", "2-4w", "1-3m"]

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

HORIZON_DEFAULT_BY_NEWS_TYPE = {
    "earnings":         "1d",
    "dividend":         "2-5d",
    "operational":      "2-5d",
    "regulatory":       "1-2w",
    "macro":            "1-2w",
    "capital_increase": "2-4w",
    "m_and_a":          "2-4w",
    "other":            "2-5d",
}

# Adjacency map -- what counts as "shifted by one bucket"
_HORIZON_ORDER = {h: i for i, h in enumerate(HORIZONS)}


# -----------------------------------------------------------------------------
#  Internal helpers
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
    if vs50 > 0 and vs200 > 0:
        return "uptrend (above MA-50 and MA-200)"
    if vs50 < 0 and vs200 < 0:
        return "downtrend (below MA-50 and MA-200)"
    if vs50 < 0 < vs200:
        return "weakening (above MA-200 but below MA-50)"
    return "recovering (above MA-50 but below MA-200)"


def _vol_interpretation(ps: dict) -> str:
    regime = ps.get("vol_regime", "N/A")
    mapping = {
        "elevated":
            "Elevated vol -> news impact AMPLIFIED; large moves more likely.",
        "normal":
            "Normal vol -> news impact PROPORTIONAL to the catalyst strength.",
        "low":
            "Low vol -> news impact MUTED unless the catalyst is structurally significant. "
            "Low vol is NOT bullish or bearish by itself.",
    }
    return mapping.get(regime, "Vol regime unknown.")


def _format_context_block(ctx: dict) -> str:
    ps  = ctx["price_state"]
    ms  = ctx["market_state"]
    cs  = ctx["correlation_state"]
    ars = ctx.get("recent_abnormal_returns", [])

    peer_lines = "\n".join(
        f"  - {p['ticker']} | sector: {p['sector']} | 90d-corr: {p['correlation']}"
        for p in cs["top_3_correlated_peers"]
    ) or "  (none)"

    ar_lines = "\n".join(
        f"  {r['date']}: stock {_v(r['stock_return_pct'])}% "
        f"| sector {_v(r['sector_return_pct'])}% "
        f"| AR {_v(r['abnormal_return_pct'])}%"
        for r in ars
    ) or "  (insufficient data)"

    ar_neg = sum(1 for r in ars if (r.get("abnormal_return_pct") or 0) < 0)
    ar_summary = f"  Summary: {ar_neg}/{len(ars)} of recent ARs are negative." if ars else ""

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
        f"Vol 20d ann : {_v(ps['vol_20d_annualized_pct'], '.2f')}%  [{ps.get('vol_regime', 'N/A')} regime]\n"
        f"Vol meaning : {_vol_interpretation(ps)}\n"
        f"RSI-14      : {_v(ps['rsi_14'], '.2f')}\n"
        f"\n"
        f"=== SECTOR / MARKET STATE ===\n"
        f"Sector return 1d   : {_v(ms['sector_return_1d'])}%\n"
        f"Sector return 5d   : {_v(ms['sector_return_5d'])}%\n"
        f"Stock vs sector 1d : {_v(ms['relative_strength_vs_sector_1d'])}%   "
        f"(negative = underperforming sector)\n"
        f"\n"
        f"=== CORRELATION ===\n"
        f"90d corr with sector index : {_v(cs['corr_with_sector_90d'], '.2f')}\n"
        f"Top correlated peers:\n"
        f"{peer_lines}\n"
        f"\n"
        f"=== RECENT ABNORMAL RETURNS (oldest -> newest) ===\n"
        f"{ar_lines}\n"
        f"{ar_summary}"
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
#  Universe guard
# -----------------------------------------------------------------------------

EGX30_UNIVERSE = {
    "ABUK", "SKPC", "AMOC", "MFPC", "EKHO",
    "COMI", "ADIB", "CIEB",
    "HRHO", "CCAP", "BTFH",
    "FWRY", "EFIH", "RAYA",
    "PHDC", "MASR", "TMGH", "EMFD", "ORHD",
    "ARCC", "MCQE", "ORAS",
    "RMDA", "ISPH",
    "JUFO", "EGAL", "ETEL", "ORWE", "EAST", "GBCO",
}


# -----------------------------------------------------------------------------
#  Public helper: build spillover candidate dict from context
# -----------------------------------------------------------------------------

def build_spillover_candidates(ctx: dict, company_descriptions: dict) -> dict:
    primary = ctx["ticker"]
    candidates = {}
    for peer in ctx["correlation_state"]["top_3_correlated_peers"]:
        t = peer["ticker"]
        if t != primary:
            candidates[t] = company_descriptions.get(t, f"{t} -- {peer['sector']}")
    return candidates


# -----------------------------------------------------------------------------
#  Horizon-selection guide (rendered into prompts)
# -----------------------------------------------------------------------------

_HORIZON_GUIDE = """\
=== HORIZON SELECTION (read before predicting) ===
There are exactly FIVE allowed horizons. Each maps to a specific future
evaluation window in trading days:

  1d     -> impact resolves by t+1   trading day  (neutral band: +-1%)
  2-5d   -> impact resolves by t+5   trading days (neutral band: +-2%)
  1-2w   -> impact resolves by t+10  trading days (neutral band: +-3%)
  2-4w   -> impact resolves by t+20  trading days (neutral band: +-4%)
  1-3m   -> impact resolves by t+60  trading days (neutral band: +-5%)

STEP 1 -- DEFAULT BY NEWS TYPE:
  earnings              -> 1d
  dividend              -> 2-5d
  operational           -> 2-5d
  regulatory            -> 1-2w
  macro                 -> 1-2w
  capital_increase      -> 2-4w
  m_and_a               -> 2-4w
  other                 -> 2-5d

STEP 2 -- TECHNICAL SHIFT (apply AT MOST ONE bucket of shift):
  Shift FASTER (one bucket shorter) IF:
    - The stock is already strongly trending in the predicted direction
      (vs MA-50 > +5% for up, < -5% for down)
    - Volatility regime is "elevated"
    - The news event happened BEFORE the article date AND the 1d return
      already moved meaningfully in the predicted direction
      (already_priced_in = true tightens the residual window)

  Shift SLOWER (one bucket longer) IF:
    - The catalyst requires regulatory approval that typically takes weeks
      (FRA approval for capital increases / mergers)
    - Volatility regime is "low" AND the catalyst is structural
    - The stock is illiquid or trending against the predicted direction

STEP 3 -- HARD RULES:
  - You may shift by AT MOST one bucket from the news-type default.
  - If already_priced_in = true, horizon must be 1d or 2-5d
    (the residual reaction window only). Do NOT use 2-4w or 1-3m for
    priced-in events.
  - The horizon you pick MUST be ONE of: 1d, 2-5d, 1-2w, 2-4w, 1-3m.
    No other strings are allowed.

STEP 4 -- JUSTIFY IN REASONING:
  In your "reasoning" field, state the news-type default and any shift
  you applied with the reason (e.g., "default for capital_increase is
  2-4w; shifted to 1-2w because already_priced_in=true and the stock
  already moved on the board-decision date").
"""


# -----------------------------------------------------------------------------
#  Shared guardrails (R1-R9)
# -----------------------------------------------------------------------------

_REASONING_GUARDRAILS = """\
=== REASONING GUARDRAILS (read carefully) ===
R1. CITE WITH LOGIC. Do not cite a context field unless it has a direct logical
    link to THIS news event. Citing "90d sector correlation = 0.77" without
    explaining why it matters for this specific news is NOT reasoning -- it is
    filler. Reject this habit.

R2. LOW VOLATILITY IS NOT BULLISH. Vol_regime is about how big the move will
    be, not its direction. Low vol + downtrend is bearish, not bullish.

R3. NEWS TYPE CHECKLIST. Before predicting, classify the news:
    - capital_increase / M&A:  ALWAYS quantify dilution = new_shares /
                               pre_existing_shares (state both inputs).
                               If dilution > 5% and no preemptive rights,
                               treat as bearish/neutral by default.
    - earnings:                 compare to consensus if mentioned; otherwise
                               compare to recent trend.
    - dividend:                 quantify yield = dividend / last_close.
                               <2% small, 2-5% medium, >5% large.
    - regulatory:               approval -> positive; probe/fine -> negative.
    - operational:              contract/win -> positive; halt/accident -> negative.
    - macro:                    company-specific impact often muted; rely on
                               sector sensitivity.

R4. TIMING CHECK. If the news describes an event that happened BEFORE the
    article date, check the recent 1d return -- the market may have ALREADY
    priced it in.

R5. PRICED-IN CHECK. If today's stock_vs_sector_1d is strongly aligned with
    the news direction, much of the impact may already be in the price.

R6. CONFIDENCE ANCHORING.
    - 0.8-1.0: clear catalyst, context agrees, no contradicting signals.
    - 0.5-0.7: clear catalyst but mixed context, OR weak catalyst with strong context.
    - 0.3-0.5: ambiguous catalyst or contradicting signals.
    - 0.0-0.3: cannot determine direction confidently.

R7. NO HALLUCINATION. Do not invent numbers. If you compute a ratio, state
    the inputs (e.g., "917M / 4,529M = 20.3%").

R8. ALREADY-PRICED-IN <-> MAGNITUDE CONSISTENCY. If already_priced_in is true,
    the FORWARD-LOOKING magnitude should typically be "small", and horizon
    should be "1d" or "2-5d" (residual window only). Setting other values
    needs explicit justification.

R9. NO LANGUAGE LEAKS. Arabic free-text fields must be 100% Arabic. No
    German, French, English, or transliterated words mid-sentence
    (e.g., "Richtung", "direction", "trend", "spillover"). Use Arabic
    equivalents (اتجاه، تأثير غير مباشر، etc.).
"""


# -----------------------------------------------------------------------------
#  Few-shot for Impact Agent
# -----------------------------------------------------------------------------

_IMPACT_FEWSHOT = """\
=== EXAMPLE (study the reasoning style, do not copy verbatim) ===
Suppose a company announces a capital increase: 200M new shares on a
1,000M pre-existing base, fully allocated to a third party, no preemptive
rights. The stock is in a downtrend (-8% over 20d, below MA-50), vol low.

Good reasoning (English here; output in Arabic):
"Classified as capital_increase. Dilution = 200M / 1,000M (pre-existing) = 20%,
fully allocated to a third party without preemptive rights -> structurally
bearish. Default horizon for capital_increase is 2-4w; kept at 2-4w because
no priced-in signal and low vol means absorption is slow. Downtrend (vs MA-50
= -X%, 20d = -8%) confirms market hasn't priced this favourably. Predicted:
down / small / 2-4w."

Bad reasoning (do NOT produce):
"Capital increase is positive. Sector correlation 0.77 means it follows the
sector. Low vol means impact will be clearer."
(Ignores dilution, cites correlation without logic, misreads low vol.)
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
You are a quantitative financial analyst specialising in the Egyptian Stock Exchange (EGX).

Your task: predict how the Arabic news article below will impact the stock price of \
{ctx['ticker']} ({ctx['company']}) over a specific time horizon.

{_format_context_block(ctx)}

=== ARABIC SENTIMENT SIGNAL (one input among many, NOT a label to defer to) ===
Pre-computed by an Arabic financial BERT model (CAMeLBERT).
Label : {sentiment_label}
Score : {_v(sentiment_score, '+.4f')}   (-1 strongly negative  ->  +1 strongly positive)
NOTE: This model classifies tone, not financial implication.

=== NEWS ARTICLE  (Arabic -- do NOT translate) ===
{_format_article(article)}

{_REASONING_GUARDRAILS}

{_HORIZON_GUIDE}

{_IMPACT_FEWSHOT}

=== OUTPUT INSTRUCTIONS ===
1. Read the Arabic article carefully. Do NOT translate.
2. Classify news_type FIRST, then predict.
3. Pick horizon using the HORIZON SELECTION rules (default + at most one shift).
4. "reasoning" MUST cite >=3 specific context fields WITH LOGIC.
5. "reasoning" MUST state the news_type default horizon and any shift applied.
6. If applicable, include a quantified ratio (dilution/yield/etc.) with inputs.
7. Magnitude rules:
   - "large": strong catalyst (>5% structural) AND elevated vol.
   - "medium": clear catalyst with normal vol.
   - "small": minor catalyst, OR low vol, OR already_priced_in=true.
8. If already_priced_in=true: magnitude defaults to "small" AND horizon
   must be "1d" or "2-5d".
9. LANGUAGE: All free-text in Arabic. JSON keys and enum values in English.
10. Output ONLY the JSON below. No markdown fences. No text outside.

{{
  "news_type":   "capital_increase" | "m_and_a" | "earnings" | "dividend" | "regulatory" | "operational" | "macro" | "other",
  "direction":   "up" | "down" | "neutral",
  "magnitude":   "small" | "medium" | "large",
  "horizon":     "1d" | "2-5d" | "1-2w" | "2-4w" | "1-3m",
  "confidence":  <float 0.0-1.0>,
  "already_priced_in": <true | false>,
  "dilution_or_yield_note": "<Arabic ratio with inputs OR 'لا ينطبق'>",
  "key_drivers": ["<driver 1 Arabic>", "<driver 2 Arabic>"],
  "reasoning":   "<3-5 sentences in Arabic. MUST cite >=3 context fields with logic. MUST state news_type default horizon and any shift applied. MUST include the quantified ratio if applicable.>"
}}"""


# -----------------------------------------------------------------------------
#  2.  SPILLOVER AGENT PROMPT
# -----------------------------------------------------------------------------

def spillover_agent_prompt(
    ctx: dict,
    article: dict,
    candidate_descriptions: dict,
) -> str:
    candidates_block = "\n".join(
        f"  - {ticker}: {desc}"
        for ticker, desc in candidate_descriptions.items()
    ) or "  (none supplied -- use your knowledge of EGX30 sector linkages)"

    corr_summary = "  " + ", ".join(
        f"{p['ticker']} (corr={p['correlation']}, {p['sector']})"
        for p in ctx["correlation_state"]["top_3_correlated_peers"]
    ) if ctx["correlation_state"]["top_3_correlated_peers"] else "  (none)"

    universe_list = ", ".join(sorted(EGX30_UNIVERSE))

    return f"""\
You are a quantitative financial analyst specialising in the Egyptian Stock Exchange (EGX).

A news event has occurred about {ctx['ticker']} ({ctx['company']}, sector: {ctx['sector']}).
Your task: identify which OTHER EGX30 companies are likely impacted by THIS SAME event,
each with its own horizon.

=== PRIMARY COMPANY ===
Ticker      : {ctx['ticker']}
Name        : {ctx['company']}
Sector      : {ctx['sector']}
Description : {ctx['business_description']}

=== NEWS ARTICLE  (Arabic -- do NOT translate) ===
{_format_article(article)}

=== STOCKS WITH HIGHEST RETURN-CORRELATION TO {ctx['ticker']} (90-day window) ===
{corr_summary}
WARNING: Correlation is statistical co-movement, not causal link.
Include a peer ONLY if you can identify an economic channel.

=== CANDIDATE COMPANIES TO EVALUATE ===
{candidates_block}

=== EGX30 UNIVERSE (allowed tickers) ===
{universe_list}
You MUST only output tickers from this list. Do NOT invent tickers.

=== SPILLOVER CHANNELS ===
  sector_comovement   shared sector news (use sparingly for company-specific events)
  supply_chain        candidate is a real supplier/customer of primary
  macro_shared        both share macro sensitivity AND news has macro implication
  competitive         direct competitor; primary's gain may be candidate's loss
  precedent           M&A/capital action signalling sector consolidation

=== SPILLOVER REASONING RULES ===
S1. EVERY spillover must have a CAUSAL channel beyond correlation.
S2. Company-specific news rarely creates large sector spillover -> default to
    fewer candidates with small magnitude.
S3. Sector / macro news creates broader spillover.
S4. Magnitude should be SMALLER than the primary impact unless the news
    is explicitly about the candidate too.
S5. If the news is idiosyncratic, return empty spillovers with no_spillover_reason.
S6. USE CORRECT ARABIC COMPANY NAMES (e.g., "بالم هيلز" for PHDC,
    "مجموعة طلعت مصطفى" for TMGH, "بلتون القابضة" for BTFH).
    Do not invent transliterations.

=== HORIZON FOR SPILLOVERS ===
Spillover horizons follow the same five-bucket scheme as the primary:
  1d, 2-5d, 1-2w, 2-4w, 1-3m
Spillover horizons are typically EQUAL TO or LONGER THAN the primary's
horizon (the market takes time to propagate the link). Pick the bucket that
best matches when the candidate will react. Do NOT use any other strings.

=== INSTRUCTIONS ===
1. Read Arabic directly; do NOT translate.
2. At most 5 spillover candidates. Fewer is better.
3. Channel must be justified by the news content.
4. LANGUAGE: All free-text in Arabic. JSON keys, tickers, enum values in English.
5. NO LANGUAGE LEAKS (per R9). 100% Arabic in reasoning fields.
6. Output ONLY the JSON below. No markdown fences. No text outside.

{{
  "news_scope": "company_specific" | "sector_wide" | "macro",
  "spillovers": [
    {{
      "ticker":    "<EGX30 ticker from universe>",
      "direction": "up" | "down" | "neutral",
      "magnitude": "small" | "medium" | "large",
      "horizon":   "1d" | "2-5d" | "1-2w" | "2-4w" | "1-3m",
      "channel":   "sector_comovement" | "supply_chain" | "macro_shared" | "competitive" | "precedent",
      "reasoning": "<1-2 sentences in Arabic explaining the CAUSAL link>"
    }}
  ],
  "no_spillover_reason": "<Arabic explanation if spillovers list is empty, else empty string>"
}}"""


# -----------------------------------------------------------------------------
#  3.  CRITIC AGENT PROMPT
# -----------------------------------------------------------------------------

def critic_agent_prompt(
    ctx: dict,
    sentiment_score: float,
    sentiment_label: str,
    article: dict,
    impact_output: dict,
    spillover_output: dict,
) -> str:
    impact_json    = json.dumps(impact_output,    ensure_ascii=False, indent=2)
    spillover_json = json.dumps(spillover_output, ensure_ascii=False, indent=2)

    return f"""\
You are a senior quantitative risk analyst at an Egyptian investment bank with deep \
expertise in EGX30 stocks, Arabic financial news, and behavioural market dynamics.

Two junior LLM analysts have independently assessed a news event about \
{ctx['ticker']} ({ctx['company']}). Your job is to:
  (a) review their work against the quantitative evidence,
  (b) catch errors (dilution, mis-cited context, timing, channel, horizon misuse),
  (c) produce the final, authoritative prediction,
  (d) write a plain-Arabic explanation for a non-expert retail investor.

{_format_context_block(ctx)}

=== ARABIC SENTIMENT SIGNAL ===
Label : {sentiment_label}
Score : {_v(sentiment_score, '+.4f')}

=== NEWS ARTICLE  (Arabic -- do NOT translate) ===
{_format_article(article)}

=== IMPACT AGENT OUTPUT (junior, Groq) ===
{impact_json}

=== SPILLOVER AGENT OUTPUT (junior, Groq) ===
{spillover_json}

{_REASONING_GUARDRAILS}

{_HORIZON_GUIDE}

=== YOUR REVIEW CHECKLIST ===
C1.  NEWS TYPE: classification correct?
C2.  QUANTIFICATION: ratio computed with conventional formula and both inputs stated?
     If missing, compute it yourself in your reasoning.
C3.  CONTEXT MISUSE: any field cited without logical link? Flag in disagreements.
C4.  TIMING / PRICED-IN: 1d return aligned with catalyst direction?
     If priced in, magnitude should be "small" and horizon "1d" or "2-5d".
C5.  TREND vs NEWS: trend (MA-50, MA-200, ARs) agree or contradict news? Disagreement lowers confidence.
C6.  VOL REGIME: any confusion of "low vol" with "bullish"?
C7.  SPILLOVER CHANNELS: causally justified, not just correlation?
C8.  SPILLOVER MAGNITUDE: small for company-specific news?
C9.  HORIZON SELECTION: did the Impact Agent pick the correct bucket?
     Check news-type default + at most one shift. Reject 2-4w/1-3m if priced in.
     Spillover horizons should be equal to or longer than the primary.
C10. INTERNAL CONSISTENCY: already_priced_in, magnitude, horizon, and confidence
     consistent with each other (R8).
C11. ARABIC COMPANY NAMES: fix hallucinated transliterations.
C12. LANGUAGE LEAKS: any non-Arabic words in Arabic fields (R9)? Strip them.

=== USER-FACING EXPLANATION (mandatory) ===
After the technical analysis, produce a plain-Arabic explanation for a
non-expert retail investor.

U1. NO JARGON. Replace technical terms with everyday Arabic:
    - "تخفيف" -> "تقليل نسبة ملكية المساهمين الحاليين"
    - "MA-50 / MA-200" -> "متوسط سعر السهم خلال الشهرين / السنة الماضية"
    - "abnormal return" -> "حركة السهم مقارنة بالسوق العامة"
    - "RSI" -> "مؤشر قوة الشراء والبيع"
    - "vol regime" -> "مستوى تذبذب السعر"
    - "spillover" -> "تأثير غير مباشر على شركات أخرى"
U2. NO NUMBERS WITHOUT MEANING. Explain what each cited number means.
U3. NO HEDGING-AS-NOISE. State the view clearly; put uncertainty in the
    dedicated field.
U4. STORY ARC: headline -> what_happened -> why_it_matters -> what_we_expect
    -> what_could_change_our_view.
U5. BREVITY. 2-3 sentences per field. ~120-180 Arabic words total.
U6. SOURCE THE CONFIDENCE. If confidence < 0.6, "what_we_expect" must reflect
    that uncertainty in tone.
U7. TONE. Professional but accessible.
U8. STATE THE PREDICTION EXPLICITLY in what_we_expect. Start with the call
    (direction + magnitude + horizon in plain Arabic), THEN add 1-2 sentences
    of supporting evidence. Do NOT bury the prediction in evidence.
U9. NO LANGUAGE LEAKS (R9). 100% Arabic.

=== INSTRUCTIONS ===
1. Read Arabic directly; do NOT translate.
2. Override juniors where evidence contradicts them. List each override in
   "disagreements_with_juniors" with the reason.
3. Confidence calibration (R6): lower when agents disagree OR when context
   contradicts direction OR when news may be priced in.
4. SPELLING: "RSI-14" not "RS-14". Proper Arabic punctuation. No stray
   ASCII artifacts.
5. LANGUAGE: All free-text in Arabic. JSON keys, tickers, enum values in English.
6. Output ONLY the JSON below. No markdown fences. No text outside.

{{
  "primary_company": {{
    "ticker":      "{ctx['ticker']}",
    "news_type":   "capital_increase" | "m_and_a" | "earnings" | "dividend" | "regulatory" | "operational" | "macro" | "other",
    "direction":   "up" | "down" | "neutral",
    "magnitude":   "small" | "medium" | "large",
    "horizon":     "1d" | "2-5d" | "1-2w" | "2-4w" | "1-3m",
    "confidence":  <float 0.0-1.0>,
    "already_priced_in": <true | false>,
    "quantified_ratio":  "<Arabic ratio with both inputs OR 'لا ينطبق'>",
    "reasoning":   "<4-6 sentences in Arabic. Cite >=3 context fields WITH logic. State news_type. State news-type default horizon and any shift applied. Include quantified ratio. Note any override.>"
  }},
  "spillovers": [
    {{
      "ticker":    "<EGX30 ticker from universe>",
      "direction": "up" | "down" | "neutral",
      "magnitude": "small" | "medium" | "large",
      "horizon":   "1d" | "2-5d" | "1-2w" | "2-4w" | "1-3m",
      "channel":   "sector_comovement" | "supply_chain" | "macro_shared" | "competitive" | "precedent",
      "reasoning": "<1-2 sentences in Arabic, causal link>"
    }}
  ],
  "risk_flags": [
    "<event-specific risk in Arabic>",
    "<event-specific risk in Arabic>"
  ],
  "disagreements_with_juniors": [
    "<which agent, what was wrong, what is correct -- in Arabic>"
  ],
  "external_signal_alignment": "aligned" | "partially_aligned" | "contradicts",
  "external_signal_explanation": "<1-2 sentences in Arabic>",
  "tldr": {{
    "verdict":   "<short Arabic phrase incl. direction + horizon>",
    "one_line":  "<one Arabic sentence summarising the whole analysis>"
  }},
  "explanation_for_user": {{
    "headline":                  "<one sentence in plain Arabic, no jargon>",
    "what_happened":             "<2-3 sentences plain Arabic, no jargon>",
    "why_it_matters":            "<2-3 sentences plain Arabic, financial logic in everyday terms>",
    "what_we_expect":            "<state the prediction FIRST (direction + size + timeline in plain Arabic), then 1-2 sentences of supporting evidence>",
    "what_could_change_our_view":"<1-2 sentences plain Arabic on key uncertainties>"
  }}
}}"""


# -----------------------------------------------------------------------------
#  Smoke test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from features import compute_context, COMPANY_DESCRIPTIONS

    ctx = compute_context("EMFD", "2025-01-02")

    dummy_article = {
        "title": "إعمار مصر تُقر زيادة رأس المال للاستحواذ على البرو نورث كوست",
        "body":  "آراب فاينانس: وافق مجلس إدارة شركة إعمار مصر...",
        "source": "Arab Finance",
        "date":   "2025-01-02",
        "datetime": "2025-01-02 10:54:00",
    }
    candidates = build_spillover_candidates(ctx, COMPANY_DESCRIPTIONS)

    print(impact_agent_prompt(ctx, 0.0579, "neutral", dummy_article))
    print("\n" + "=" * 70 + "\n")
    print(spillover_agent_prompt(ctx, dummy_article, candidates))