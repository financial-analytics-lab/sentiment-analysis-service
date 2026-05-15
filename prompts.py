# prompts.py
"""
Prompt templates for the three-agent EGX30 news-impact pipeline.

    impact_agent_prompt()    -> str   for Groq (Qwen3-32B or similar)
    spillover_agent_prompt() -> str   for Groq (Qwen3-32B or similar)
    critic_agent_prompt()    -> str   for Claude (Anthropic)

Rules enforced by every prompt:
  - Arabic news is kept in Arabic -- never translated.
  - Instructions are in English.
  - Models must output ONLY valid JSON (no markdown fences, no surrounding prose).
"""

import json


# -----------------------------------------------------------------------------
#  Internal helpers
# -----------------------------------------------------------------------------

def _v(val, fmt: str = "+.2f", fallback: str = "N/A") -> str:
    """Format a possibly-None number. Uses fallback when val is None."""
    if val is None:
        return fallback
    try:
        return format(val, fmt)
    except (TypeError, ValueError):
        return str(val)


def _trend_label(ps: dict) -> str:
    """Compact human-readable trend label derived from price_state."""
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
    """Tell the model what vol_regime actually means for impact magnitude."""
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
    """Render the compute_context() dict as a compact, readable prompt section."""
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
    """Render an article dict as a prompt section (Arabic body kept as-is)."""
    return (
        f"Source    : {article.get('source', 'N/A')}\n"
        f"Published : {article.get('datetime') or article.get('date', 'N/A')}\n"
        f"Headline  : {article.get('title', '')}\n"
        f"\n"
        f"{article.get('body') or article.get('title', '')}"
    )


# -----------------------------------------------------------------------------
#  Universe guard -- keep tickers honest in spillover output
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
    """
    Build {ticker: description} for the Spillover Agent using the top
    correlated peers already stored in the context dict.

    Pass COMPANY_DESCRIPTIONS from features.py as company_descriptions.
    """
    primary = ctx["ticker"]
    candidates = {}
    for peer in ctx["correlation_state"]["top_3_correlated_peers"]:
        t = peer["ticker"]
        if t != primary:
            candidates[t] = company_descriptions.get(t, f"{t} -- {peer['sector']}")
    return candidates


# -----------------------------------------------------------------------------
#  Shared "guardrails" block reused in Impact and Critic prompts
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
    - capital_increase / M&A:  ALWAYS quantify dilution. Use the CONVENTIONAL
                               definition: dilution = new_shares / pre_existing_shares
                               (NOT new_shares / total_post_increase). State both
                               the numerator and denominator explicitly.
                               If dilution > 5% and existing shareholders have no
                               preemptive rights, treat as bearish/neutral by default
                               unless the strategic rationale clearly outweighs it.
    - earnings:                 compare to consensus if mentioned; otherwise compare
                               to recent trend.
    - dividend:                 quantify yield = dividend / last_close. <2% = small,
                               2-5% = medium, >5% = large.
    - regulatory:               regulatory approval -> positive; regulatory probe
                               or fine -> negative.
    - operational:              new contract / project win -> positive; production
                               halt / accident -> negative.
    - macro:                    company-specific impact often muted; rely on sector
                               sensitivity.

R4. TIMING CHECK. If the news describes an event that happened BEFORE the article
    date (e.g., a board decision from yesterday published today), check the recent
    1d return -- the market may have ALREADY priced it in.

R5. PRICED-IN CHECK. If today's stock_vs_sector_1d is strongly positive AND the
    news is positive, much of the impact may already be in the price.

R6. CONFIDENCE ANCHORING.
    - 0.8-1.0: clear catalyst, context agrees, no contradicting signals.
    - 0.5-0.7: clear catalyst but mixed context, OR weak catalyst with strong context.
    - 0.3-0.5: ambiguous catalyst or contradicting signals.
    - 0.0-0.3: cannot determine direction confidently.

R7. NO HALLUCINATION. Do not invent numbers. If the article gives a number
    (e.g., capital increase amount), use it. If you compute a ratio, state the
    inputs (e.g., "917M new shares / 4,529M pre-existing = 20.3% dilution").
    

R8. ALREADY-PRICED-IN <-> MAGNITUDE CONSISTENCY. If already_priced_in is true,
    the FORWARD-LOOKING magnitude should typically be "small", because the
    priced-in portion is by definition behind us. Setting already_priced_in=true
    with magnitude="medium" or "large" needs explicit justification (e.g., the
    move so far covers only part of the catalyst).
    
R9. NO LANGUAGE LEAKS. Arabic free-text fields must be 100% Arabic.
   No German, French, or English words inside Arabic sentences (e.g.,
   "Richtung", "direction", "trend"). If you need a technical term,
   use the Arabic equivalent: اتجاه instead of "direction" or "Richtung".
    """
    


# -----------------------------------------------------------------------------
#  Few-shot anchor for the Impact Agent
# -----------------------------------------------------------------------------

_IMPACT_FEWSHOT = """\
=== EXAMPLE (study the reasoning style, do not copy verbatim) ===
Suppose a company announces a capital increase: 200M new shares on a 1,000M
pre-existing base, fully allocated to a third party, no preemptive rights.
The stock is in a downtrend (-8% over 20d, below MA-50) and vol is low.

Good reasoning (Arabic in real output; English here for clarity):
"Classified as capital_increase. Dilution = 200M / 1,000M (pre-existing) = 20%,
fully allocated to a third party without preemptive rights -> structurally
bearish for existing shareholders. The downtrend (vs MA-50 = -X%, 20d return
= -8%) confirms the market has not priced this favorably. Low vol means the
move will be muted in the short term but persistent. Predicted: down / small
/ 1-4w."

Bad reasoning (do NOT produce this style):
"The news is about a capital increase which is positive for the company.
Sector correlation is 0.77 so the company will follow the sector. Volatility
is low so the impact will be clearer."
(Why bad: ignores dilution, cites correlation without logic, misreads low vol.)
"""


# -----------------------------------------------------------------------------
#  1.  IMPACT AGENT PROMPT  (Groq -- Qwen3-32B / Llama 3.3 70B)
# -----------------------------------------------------------------------------

def impact_agent_prompt(
    ctx: dict,
    sentiment_score: float,
    sentiment_label: str,
    article: dict,
) -> str:
    """
    Prompt for the Impact Agent.
    Predicts direction, magnitude, and horizon for the PRIMARY company only.
    """
    return f"""\
You are a quantitative financial analyst specialising in the Egyptian Stock Exchange (EGX).

Your task: predict how the Arabic news article below will impact the stock price of \
{ctx['ticker']} ({ctx['company']}) over the next days to weeks.

{_format_context_block(ctx)}

=== ARABIC SENTIMENT SIGNAL (one input among many, NOT a label to defer to) ===
Pre-computed by an Arabic financial BERT model (CAMeLBERT).
Label : {sentiment_label}
Score : {_v(sentiment_score, '+.4f')}   (-1 = strongly negative  ->  +1 = strongly positive)
NOTE: This model classifies tone, not financial implication. A news article can
sound "neutral" in tone but be financially bearish (e.g., dilution), or sound
"positive" but be financially neutral (e.g., a routine announcement). Use this
score as a sanity check, not as ground truth.

=== NEWS ARTICLE  (Arabic -- do NOT translate) ===
{_format_article(article)}

{_REASONING_GUARDRAILS}

{_IMPACT_FEWSHOT}

=== OUTPUT INSTRUCTIONS ===
1. Read the Arabic article carefully. Do NOT translate; reason from the Arabic directly.
2. Classify the news_type FIRST (per R3), then predict.
3. Your "reasoning" field MUST cite at least THREE specific context fields by name
   WITH LOGIC (per R1). Generic citations are rejected.
4. If applicable, INCLUDE A QUANTIFIED RATIO in reasoning (dilution %, yield %, etc.).
   Use the CONVENTIONAL dilution formula (new shares / pre-existing shares).
5. Horizon guidance:
   - earnings/dividend surprises: "1d" or "2-5d"
   - operational news (contract, halt): "2-5d"
   - structural/regulatory/M&A: "1-4w"
6. Magnitude guidance:
   - "large" requires BOTH a strong catalyst (>5% structural change) AND elevated vol
   - "medium" = clear catalyst with normal vol
   - "small" = minor catalyst, OR low vol regardless of catalyst strength,
                OR already_priced_in=true
7. If already_priced_in=true, magnitude should default to "small" (per R8).
8. LANGUAGE: All free-text values (reasoning, key_drivers items, dilution_or_yield_note)
   MUST be written in Arabic. JSON keys and enum values stay in English.
9. Output ONLY the JSON object below. No markdown fences. No text outside the JSON.

{{
  "news_type":   "capital_increase" | "m_and_a" | "earnings" | "dividend" | "regulatory" | "operational" | "macro" | "other",
  "direction":   "up" | "down" | "neutral",
  "magnitude":   "small" | "medium" | "large",
  "horizon":     "1d" | "2-5d" | "1-4w",
  "confidence":  <float 0.0-1.0>,
  "already_priced_in": <true | false>,
  "dilution_or_yield_note": "<Arabic: e.g. 'تخفيف = 917م / 4,529م (قبل الزيادة) = 20.3%'  OR  'لا ينطبق'>",
  "key_drivers": ["<driver 1 in Arabic>", "<driver 2 in Arabic>"],
  "reasoning":   "<3-5 sentences in Arabic. MUST cite >=3 context fields with logic. MUST mention news_type. MUST include any computed ratio.>"
}}"""


# -----------------------------------------------------------------------------
#  2.  SPILLOVER AGENT PROMPT  (Groq -- Qwen3-32B / Llama 3.3 70B)
# -----------------------------------------------------------------------------

def spillover_agent_prompt(
    ctx: dict,
    article: dict,
    candidate_descriptions: dict,
) -> str:
    """
    Prompt for the Spillover Agent.
    Identifies which OTHER EGX30 companies are affected by the news event.
    """
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
Your task: identify which OTHER EGX30 companies are likely impacted by THIS SAME event.

=== PRIMARY COMPANY ===
Ticker      : {ctx['ticker']}
Name        : {ctx['company']}
Sector      : {ctx['sector']}
Description : {ctx['business_description']}

=== NEWS ARTICLE  (Arabic -- do NOT translate) ===
{_format_article(article)}

=== STOCKS WITH HIGHEST RETURN-CORRELATION TO {ctx['ticker']} (90-day window) ===
{corr_summary}
WARNING: High correlation is a STATISTICAL co-movement signal, not a causal link.
Do NOT include a peer in spillovers JUST because it correlates. You must identify
an ECONOMIC channel below.

=== CANDIDATE COMPANIES TO EVALUATE (suggested -- you may add others from the universe) ===
{candidates_block}

=== EGX30 UNIVERSE (allowed tickers) ===
{universe_list}
You MUST only output tickers from this list. Do NOT invent tickers.

=== SPILLOVER CHANNELS -- pick the one that fits best ===
  sector_comovement   stocks in the same sector move together on shared sector news
                      (use sparingly -- company-specific news rarely moves the whole sector)
  supply_chain        the candidate is a real supplier or customer of the primary company
  macro_shared        both share the same macro sensitivity (e.g., natural gas, EGP/USD,
                      interest rates) AND this news has a macro implication
  competitive         direct competitor -- a gain for the primary may be a loss for the
                      candidate (or vice versa). Most relevant for market-share news,
                      regulatory wins/losses, and M&A signaling.
  precedent           M&A or capital action that signals likely consolidation/dilution
                      in peers -- candidate may re-rate on takeout speculation.

=== SPILLOVER REASONING RULES ===
S1. EVERY spillover must have a CAUSAL channel beyond raw correlation.
    State the cause in one sentence, not "they are in the same sector".
S2. COMPANY-SPECIFIC news (earnings beat, single-firm capital action, single-firm
    regulatory issue) rarely creates large sector spillover. Default to fewer
    candidates with small magnitude.
S3. SECTOR / MACRO news (regulation affecting the whole sector, FX move, commodity
    price shock) creates larger and broader spillover.
S4. Magnitude should be SMALLER than the primary impact unless the news is
    explicitly about the candidate too.
S5. If the news is purely idiosyncratic, return an empty list with "no_spillover_reason".
S6. USE CORRECT ARABIC COMPANY NAMES (e.g., "بالم هيلز" for PHDC, "تي إم جي القابضة"
    or "مجموعة طلعت مصطفى" for TMGH, "بلتون القابضة" for BTFH). Do not invent
    transliterations.

=== INSTRUCTIONS ===
1. Read the Arabic article directly; do NOT translate.
2. Include at most 5 spillover candidates. Fewer is better; weak links add noise.
3. For each candidate, the "channel" must be justified by the news content,
   not just by correlation or sector membership.
4. LANGUAGE: All free-text values (reasoning, no_spillover_reason) MUST be in Arabic.
   JSON keys, tickers, and enum values stay in English.
5. Output ONLY the JSON object below. No markdown fences. No text outside the JSON.

{{
  "news_scope": "company_specific" | "sector_wide" | "macro",
  "spillovers": [
    {{
      "ticker":    "<EGX30 ticker from universe>",
      "direction": "up" | "down" | "neutral",
      "magnitude": "small" | "medium" | "large",
      "channel":   "sector_comovement" | "supply_chain" | "macro_shared" | "competitive" | "precedent",
      "reasoning": "<1-2 sentences in Arabic explaining the CAUSAL link (not just correlation)>"
    }}
  ],
  "no_spillover_reason": "<Arabic explanation if spillovers list is empty, else empty string>"
}}"""


# -----------------------------------------------------------------------------
#  3.  CRITIC AGENT PROMPT  (Claude -- Anthropic API)
# -----------------------------------------------------------------------------

def critic_agent_prompt(
    ctx: dict,
    sentiment_score: float,
    sentiment_label: str,
    article: dict,
    impact_output: dict,
    spillover_output: dict,
) -> str:
    """
    Prompt for the Critic Agent (Claude).
    Reviews both junior outputs and produces the final authoritative prediction,
    plus a plain-Arabic explanation for non-expert readers.
    """
    impact_json    = json.dumps(impact_output,    ensure_ascii=False, indent=2)
    spillover_json = json.dumps(spillover_output, ensure_ascii=False, indent=2)

    return f"""\
You are a senior quantitative risk analyst at an Egyptian investment bank with \
deep expertise in EGX30 stocks, Arabic financial news, and behavioural market dynamics.

Two junior LLM analysts have independently assessed a news event about \
{ctx['ticker']} ({ctx['company']}). Your job is to:
  (a) critically review their work against the hard quantitative evidence,
  (b) catch errors the juniors missed (dilution, mis-cited context, timing, channel misuse),
  (c) produce the final, authoritative prediction,
  (d) write a plain-Arabic explanation that a non-expert retail investor can understand.

{_format_context_block(ctx)}

=== ARABIC SENTIMENT SIGNAL ===
Label : {sentiment_label}
Score : {_v(sentiment_score, '+.4f')}   (-1 = strongly negative  ->  +1 = strongly positive)

=== NEWS ARTICLE  (Arabic -- do NOT translate) ===
{_format_article(article)}

=== IMPACT AGENT OUTPUT (junior, Groq) ===
{impact_json}

=== SPILLOVER AGENT OUTPUT (junior, Groq) ===
{spillover_json}

{_REASONING_GUARDRAILS}

=== YOUR REVIEW CHECKLIST ===
C1. NEWS TYPE: Did the Impact Agent classify news_type correctly?
C2. QUANTIFICATION: For M&A / capital increases / dividends, did the Impact Agent
    compute and cite the relevant ratio using the CONVENTIONAL formula
    (new shares / pre-existing shares for dilution)? If not, COMPUTE IT YOURSELF
    and include it in your reasoning. State both inputs explicitly.
C3. CONTEXT MISUSE: Did the Impact Agent cite context fields WITHOUT logical link
    (e.g., correlation as a directional signal)? Flag this in disagreements_with_juniors.
C4. TIMING: Could this news already be priced in? Check the 1d return against the
    expected direction of the catalyst. If priced in, magnitude should be "small"
    (per R8) unless explicitly justified otherwise.
C5. TREND vs NEWS: Does the existing trend (MA-50, MA-200, recent ARs) agree with
    or contradict the news direction? Disagreement should lower confidence.
C6. VOL REGIME: Did the Impact Agent confuse "low vol" with "bullish"? Flag it.
C7. SPILLOVER CHANNELS: Are the spillover channels causally justified? Reject any
    spillover whose only justification is correlation or sector membership.
C8. SPILLOVER MAGNITUDE: For company-specific news, spillover magnitudes should be
    SMALL. Downgrade any "medium"/"large" that the Spillover Agent over-rated.
C9. MISSING RISKS: Identify event-specific risks neither agent mentioned (regulatory
    approval timelines, liquidity at this volume regime, earnings calendar
    proximity, FX exposure relevant to THIS company, etc.).
    GENERIC risk flags ("volatility could amplify") are REJECTED -- be specific.
C10. INTERNAL CONSISTENCY: Check that already_priced_in, magnitude, and confidence
     are logically consistent with each other (per R8).
C11. ARABIC COMPANY NAMES: Verify the junior used correct Arabic names. Fix any
     hallucinated transliterations (e.g., "بركات هيلز" should be "بالم هيلز").

=== USER-FACING EXPLANATION (mandatory) ===
After the technical analysis, produce a plain-Arabic explanation for a non-expert
reader (educated Egyptian retail investor, NOT a finance professional).

Rules for this section:
U1. NO JARGON. Replace technical terms with everyday Arabic equivalents:
    - "تخفيف" -> "تقليل نسبة ملكية المساهمين الحاليين"
    - "MA-50" / "MA-200" -> "متوسط سعر السهم خلال الشهرين أو السنة الماضية"
    - "abnormal return" -> "حركة السهم مقارنة بحركة السوق العامة"
    - "RSI" -> "مؤشر قوة الشراء والبيع"
    - "vol regime" -> "مستوى تذبذب السعر"
    - "spillover" -> "تأثير غير مباشر على شركات أخرى"
U2. NO NUMBERS WITHOUT MEANING. If you cite a number, explain what it means.
    Bad: "السهم تحت MA-50 بـ5.62%"
    Good: "السهم يتداول تحت متوسط سعره في الشهرين الماضيين، مما يعني أن السوق
           متشائم تجاهه حاليًا"
U3. NO HEDGING-AS-NOISE. Don't say "may, might, could, possibly" five times in
    one paragraph. State the view clearly, then state uncertainty separately
    in "what_could_change_our_view".
U4. STORY ARC. The fields should flow as a story:
    headline -> what_happened -> why_it_matters -> what_we_expect -> what_could_change_our_view.
U5. BREVITY. Each field 2-3 sentences max. Total length around 120-180 Arabic words.
U6. SOURCE THE CONFIDENCE. If confidence is below 0.6, the "what_we_expect" field
    must reflect uncertainty in tone, not state the prediction as certain.
U7. TONE. Professional but accessible. Imagine explaining to a smart friend who
    invests his own savings and doesn't have a finance background.
U8. STATE THE PREDICTION EXPLICITLY in what_we_expect.
    Start with the call ("نتوقع ضغوط هبوطية محدودة على السهم خلال 2-5 أيام"),
    THEN add 1-2 sentences of supporting evidence.
    Do not bury the prediction in evidence.

=== INSTRUCTIONS ===
1. Read the Arabic article directly; do NOT translate.
2. Override the junior agents where the quantitative evidence clearly contradicts them.
   Each override must be listed in "disagreements_with_juniors" with the reason.
3. "confidence" calibration (per R6): lower it when agents disagree OR when context
   contradicts the news direction OR when news may be priced in.
4. SPELLING: Use "RSI-14" not "RS-14". Use proper Arabic punctuation. No stray
   ASCII artifacts inside Arabic text.
5. LANGUAGE: All free-text values MUST be in Arabic. JSON keys, tickers, and enum
   values stay in English.
6. Output ONLY the JSON object below. No markdown fences. No text outside the JSON.
7. NO LANGUAGE LEAKS. Arabic free-text fields must be 100% Arabic.
   No German, French, or English words inside Arabic sentences (e.g.,
   "Richtung", "direction", "trend"). If you need a technical term,
   use the Arabic equivalent: اتجاه instead of "direction" or "Richtung".

{{
  "primary_company": {{
    "ticker":      "{ctx['ticker']}",
    "news_type":   "capital_increase" | "m_and_a" | "earnings" | "dividend" | "regulatory" | "operational" | "macro" | "other",
    "direction":   "up" | "down" | "neutral",
    "magnitude":   "small" | "medium" | "large",
    "horizon":     "1d" | "2-5d" | "1-4w",
    "confidence":  <float 0.0-1.0>,
    "already_priced_in": <true | false>,
    "quantified_ratio":  "<Arabic: e.g. 'تخفيف = 917م / 4,529م (قبل الزيادة) ≈ 20.3%' OR 'لا ينطبق'>",
    "reasoning":   "<4-6 sentences in Arabic. MUST cite >=3 context fields WITH logic. MUST state news_type. MUST include the quantified ratio when applicable. MUST note any override of the juniors.>"
  }},
  "spillovers": [
    {{
      "ticker":    "<EGX30 ticker from the allowed universe>",
      "direction": "up" | "down" | "neutral",
      "magnitude": "small" | "medium" | "large",
      "channel":   "sector_comovement" | "supply_chain" | "macro_shared" | "competitive" | "precedent",
      "reasoning": "<1-2 sentences in Arabic with CAUSAL link, not correlation>"
    }}
  ],
  "risk_flags": [
    "<event-specific risk in Arabic>",
    "<event-specific risk in Arabic>"
  ],
  "disagreements_with_juniors": [
    "<specific override in Arabic: which agent, what was wrong, what is correct>"
  ],
  "external_signal_alignment": "aligned" | "partially_aligned" | "contradicts",
  "external_signal_explanation": "<1-2 sentences in Arabic on whether sentiment score agrees with your direction and why>",
  "tldr": {{
    "verdict":   "<short Arabic phrase: e.g. 'هابط لمدة 1-4 أسابيع' or 'صاعد قصير الأجل' or 'محايد'>",
    "one_line":  "<one Arabic sentence summarising the entire analysis>"
  }},
  "explanation_for_user": {{
    "headline":                  "<one sentence in plain Arabic, no jargon>",
    "what_happened":             "<2-3 sentences in plain Arabic explaining the news without jargon>",
    "why_it_matters":            "<2-3 sentences in plain Arabic explaining the financial logic in everyday terms>",
    "what_we_expect":            "<2-3 sentences in plain Arabic on direction + approximate size + timeline>",
    "what_could_change_our_view":"<1-2 sentences in plain Arabic on key uncertainties that could flip the prediction>"
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

    ctx = compute_context("ABUK", "2025-01-04")

    dummy_article = {
        "title": "أبو قير للأسمدة تُقر توزيع كوبون نقدي على المساهمين",
        "body":  "أعلنت شركة أبو قير للأسمدة عن توزيع كوبون نقدي قدره 2.30 جنيه لكل سهم.",
        "source": "Arab Finance",
        "date":   "2025-01-05",
        "datetime": "2025-01-05 10:54:00",
    }

    candidates = build_spillover_candidates(ctx, COMPANY_DESCRIPTIONS)

    print("=" * 70); print("IMPACT AGENT PROMPT"); print("=" * 70)
    print(impact_agent_prompt(ctx, 0.72, "positive", dummy_article))

    print("\n" + "=" * 70); print("SPILLOVER AGENT PROMPT"); print("=" * 70)
    print(spillover_agent_prompt(ctx, dummy_article, candidates))

    print("\n" + "=" * 70); print("CRITIC AGENT PROMPT  (with dummy junior outputs)"); print("=" * 70)
    dummy_impact   = {
        "news_type": "dividend", "direction": "up", "magnitude": "small", "horizon": "2-5d",
        "confidence": 0.7, "already_priced_in": False,
        "dilution_or_yield_note": "العائد = 2.30 / سعر الإغلاق",
        "key_drivers": ["إعلان كوبون نقدي"],
        "reasoning": "خبر توزيع كوبون نقدي إيجابي."
    }
    dummy_spillover = {
        "news_scope": "company_specific",
        "spillovers": [
            {"ticker": "MFPC", "direction": "neutral", "magnitude": "small",
             "channel": "sector_comovement",
             "reasoning": "نفس القطاع، أثر ضعيف."}
        ],
        "no_spillover_reason": ""
    }
    print(critic_agent_prompt(ctx, 0.72, "positive", dummy_article, dummy_impact, dummy_spillover))