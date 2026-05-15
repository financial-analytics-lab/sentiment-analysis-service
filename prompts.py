# prompts.py
"""
Prompt templates for the three-agent EGX30 news-impact pipeline.

    impact_agent_prompt()    -> str   for Groq Llama 3.3 70B
    spillover_agent_prompt() -> str   for Groq Llama 3.3 70B
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
        f"Vol 20d ann : {_v(ps['vol_20d_annualized_pct'], '.2f')}%  [{ps.get('vol_regime', 'N/A')} regime]\n"
        f"RSI-14      : {_v(ps['rsi_14'], '.2f')}\n"
        f"\n"
        f"=== SECTOR STATE ===\n"
        f"Sector return 1d   : {_v(ms['sector_return_1d'])}%\n"
        f"Sector return 5d   : {_v(ms['sector_return_5d'])}%\n"
        f"Stock vs sector 1d : {_v(ms['relative_strength_vs_sector_1d'])}%\n"
        f"\n"
        f"=== CORRELATION ===\n"
        f"90d corr with sector index : {_v(cs['corr_with_sector_90d'], '.2f')}\n"
        f"Top correlated peers:\n"
        f"{peer_lines}\n"
        f"\n"
        f"=== RECENT ABNORMAL RETURNS (oldest -> newest) ===\n"
        f"{ar_lines}"
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
#  Public helper: build spillover candidate dict from context
# -----------------------------------------------------------------------------

def build_spillover_candidates(ctx: dict, company_descriptions: dict) -> dict:
    """
    Build {ticker: description} for the Spillover Agent using the top
    correlated peers already stored in the context dict.

    Pass COMPANY_DESCRIPTIONS from features.py as company_descriptions.
    """
    candidates = {}
    for peer in ctx["correlation_state"]["top_3_correlated_peers"]:
        t = peer["ticker"]
        candidates[t] = company_descriptions.get(t, f"{t} -- {peer['sector']}")
    return candidates


# -----------------------------------------------------------------------------
#  1.  IMPACT AGENT PROMPT  (Groq -- Llama 3.3 70B)
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

=== ARABIC SENTIMENT SIGNAL ===
Pre-computed by an Arabic financial BERT model (CAMeLBERT).
Label : {sentiment_label}
Score : {_v(sentiment_score, '+.4f')}   (-1 = strongly negative  ->  +1 = strongly positive)

=== NEWS ARTICLE  (Arabic -- do NOT translate) ===
{_format_article(article)}

=== INSTRUCTIONS ===
1. Read the Arabic article carefully. Do NOT translate it; reason from it directly.
2. Cross-reference the news with the quantitative context above.
3. Your "reasoning" field MUST cite at least THREE specific context fields by name
   (e.g. "the 60d return of -21.27% shows the stock is already in a downtrend, so ...").
4. The Arabic sentiment score is one signal among many -- do not let it override clear
   fundamental evidence in the quantitative context.
5. horizon guidance: structural/regulatory news -> "1-4w"; earnings surprises -> "1d"/"2-5d".
6. magnitude guidance: "large" requires both a strong news catalyst AND elevated volatility.
7. LANGUAGE: All free-text values (reasoning, key_drivers items) MUST be written in Arabic.
   JSON keys and enum values (direction/magnitude/horizon) stay in English.
8. Output ONLY the JSON object below. No markdown fences. No text outside the JSON.

{{
  "direction":   "up" | "down" | "neutral",
  "magnitude":   "small" | "medium" | "large",
  "horizon":     "1d" | "2-5d" | "1-4w",
  "confidence":  <float 0.0-1.0>,
  "key_drivers": ["<driver 1 in Arabic>", "<driver 2 in Arabic>"],
  "reasoning":   "<2-4 sentences in Arabic citing specific context field values>"
}}"""


# -----------------------------------------------------------------------------
#  2.  SPILLOVER AGENT PROMPT  (Groq -- Llama 3.3 70B)
# -----------------------------------------------------------------------------

def spillover_agent_prompt(
    ctx: dict,
    article: dict,
    candidate_descriptions: dict,
) -> str:
    """
    Prompt for the Spillover Agent.
    Identifies which OTHER EGX30 companies are affected by the news event.

    candidate_descriptions: {ticker: one-line description}
    Use build_spillover_candidates() to construct this from ctx.
    """
    candidates_block = "\n".join(
        f"  - {ticker}: {desc}"
        for ticker, desc in candidate_descriptions.items()
    ) or "  (none supplied -- use your knowledge of EGX30 sector linkages)"

    corr_summary = "  " + ", ".join(
        f"{p['ticker']} (corr={p['correlation']}, {p['sector']})"
        for p in ctx["correlation_state"]["top_3_correlated_peers"]
    ) if ctx["correlation_state"]["top_3_correlated_peers"] else "  (none)"

    return f"""\
You are a quantitative financial analyst specialising in the Egyptian Stock Exchange (EGX).

A news event has occurred about {ctx['ticker']} ({ctx['company']}, sector: {ctx['sector']}).
Your task: identify which OTHER EGX30 companies are likely to be impacted by this same event.

=== PRIMARY COMPANY ===
Ticker      : {ctx['ticker']}
Name        : {ctx['company']}
Sector      : {ctx['sector']}
Description : {ctx['business_description']}

=== NEWS ARTICLE  (Arabic -- do NOT translate) ===
{_format_article(article)}

=== STOCKS WITH HIGHEST RETURN-CORRELATION TO {ctx['ticker']} (90-day window) ===
{corr_summary}

=== CANDIDATE COMPANIES TO EVALUATE ===
{candidates_block}

=== SPILLOVER CHANNELS -- pick the best fit ===
  sector_comovement  stocks in the same sector move together on shared macro news
  supply_chain       this company is a key supplier or customer of the candidate
  macro_shared       both share the same sensitivity (e.g. natural gas price, EGP/USD)
  competitive        direct competitor -- a gain for one may be a loss for the other

=== INSTRUCTIONS ===
1. Read the Arabic article directly; do NOT translate.
2. For each candidate you include, state a clear causal link to the news event.
3. Include at most 5 spillover candidates. Fewer is better; weak links add noise.
4. If no other company is meaningfully affected, return an empty list.
5. LANGUAGE: All free-text values (reasoning) MUST be written in Arabic.
   JSON keys and enum values (ticker/direction/magnitude/channel) stay in English.
6. Output ONLY the JSON object below. No markdown fences. No text outside the JSON.

{{
  "spillovers": [
    {{
      "ticker":    "<EGX ticker>",
      "direction": "up" | "down" | "neutral",
      "magnitude": "small" | "medium" | "large",
      "channel":   "sector_comovement" | "supply_chain" | "macro_shared" | "competitive",
      "reasoning": "<1-2 sentences in Arabic explaining the causal link to the news>"
    }}
  ]
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
    Reviews both junior outputs and produces the final authoritative prediction.
    """
    impact_json   = json.dumps(impact_output,    ensure_ascii=False, indent=2)
    spillover_json = json.dumps(spillover_output, ensure_ascii=False, indent=2)

    return f"""\
You are a senior quantitative risk analyst at an Egyptian investment bank with \
deep expertise in EGX30 stocks, Arabic financial news, and behavioural market dynamics.

Two junior LLM analysts have independently assessed a news event about \
{ctx['ticker']} ({ctx['company']}). Your job is to:
  (a) critically review their work against the hard quantitative evidence,
  (b) resolve disagreements between them,
  (c) produce the final, authoritative prediction.

{_format_context_block(ctx)}

=== ARABIC SENTIMENT SIGNAL ===
Label : {sentiment_label}
Score : {_v(sentiment_score, '+.4f')}   (-1 = strongly negative  ->  +1 = strongly positive)

=== NEWS ARTICLE  (Arabic -- do NOT translate) ===
{_format_article(article)}

=== IMPACT AGENT OUTPUT (Llama 3.3 70B) ===
{impact_json}

=== SPILLOVER AGENT OUTPUT (Llama 3.3 70B) ===
{spillover_json}

=== YOUR REVIEW CHECKLIST ===
1. Is the Impact Agent's direction consistent with the article AND the quantitative context?
   If they contradict each other, which is more credible and why?
2. Does the Impact Agent cite the right context fields? Did it ignore important signals
   (e.g., vol regime, trend vs MA-50/MA-200, recent abnormal returns)?
3. Are the Spillover Agent's channels plausible? Are any spillovers missing or spurious?
4. Does the Arabic sentiment score align with the fundamental read? Flag if it conflicts.
5. Identify any risk factors neither agent mentioned
   (liquidity, regulatory tail risk, sector headwinds, earnings calendar proximity, etc.).

=== INSTRUCTIONS ===
1. Read the Arabic article directly; do NOT translate.
2. Override the junior agents where the quantitative evidence clearly contradicts them --
   explain the override in your reasoning.
3. "confidence" should be lower when agents disagree or when context signals are mixed.
4. LANGUAGE: All free-text values (reasoning, risk_flags items, disagreements_with_juniors items)
   MUST be written in Arabic. JSON keys and enum values stay in English.
5. Output ONLY the JSON object below. No markdown fences. No text outside the JSON.

{{
  "primary_company": {{
    "ticker":     "{ctx['ticker']}",
    "direction":  "up" | "down" | "neutral",
    "magnitude":  "small" | "medium" | "large",
    "horizon":    "1d" | "2-5d" | "1-4w",
    "confidence": <float 0.0-1.0>,
    "reasoning":  "<3-5 sentences in Arabic citing context field values; note any overrides>"
  }},
  "spillovers": [
    {{
      "ticker":    "<EGX ticker>",
      "direction": "up" | "down" | "neutral",
      "magnitude": "small" | "medium" | "large",
      "channel":   "sector_comovement" | "supply_chain" | "macro_shared" | "competitive",
      "reasoning": "<1-2 sentences in Arabic>"
    }}
  ],
  "risk_flags":                 ["<flag in Arabic>", "<flag in Arabic>"],
  "disagreements_with_juniors": ["<description in Arabic>"],
  "external_signal_alignment":  "aligned" | "partially_aligned" | "contradicts"
}}"""


# -----------------------------------------------------------------------------
#  Smoke test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from features import compute_context, COMPANY_DESCRIPTIONS

    ctx = compute_context("ABUK", "2025-01-04")   # day before article

    dummy_article = {
        "title": "أبو قير للأسمدة تُقر توزيع كوبون نقدي على المساهمين",
        "body":  "أعلنت شركة أبو قير للأسمدة عن توزيع كوبون نقدي قدره 2.30 جنيه لكل سهم.",
        "source": "Arab Finance",
        "date":   "2025-01-05",
        "datetime": "2025-01-05 10:54:00",
    }

    candidates = build_spillover_candidates(ctx, COMPANY_DESCRIPTIONS)

    print("=" * 70)
    print("IMPACT AGENT PROMPT")
    print("=" * 70)
    print(impact_agent_prompt(ctx, sentiment_score=0.72, sentiment_label="positive", article=dummy_article))

    print("\n" + "=" * 70)
    print("SPILLOVER AGENT PROMPT")
    print("=" * 70)
    print(spillover_agent_prompt(ctx, dummy_article, candidates))

    print("\n" + "=" * 70)
    print("CRITIC AGENT PROMPT  (with dummy junior outputs)")
    print("=" * 70)
    dummy_impact   = {"direction": "up", "magnitude": "small", "horizon": "2-5d",
                      "confidence": 0.7, "key_drivers": ["dividend announcement"],
                      "reasoning": "Dividend news is positive."}
    dummy_spillover = {"spillovers": [
        {"ticker": "MFPC", "direction": "neutral", "magnitude": "small",
         "channel": "sector_comovement", "reasoning": "Same sector, weak link."}
    ]}
    print(critic_agent_prompt(ctx, 0.72, "positive", dummy_article, dummy_impact, dummy_spillover))
