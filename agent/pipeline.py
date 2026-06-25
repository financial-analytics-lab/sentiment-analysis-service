"""
Main pipeline orchestrator.

Execution order:
  Tier-0 (sequential, ~15 ms):  build_context
  Tier-1 (parallel, ~200-400 ms): run_expert || run_sentiment
  Tier-2 (sequential, ~400-1200 ms): run_analyst
  Gate   (<1 ms, deterministic)
  Tier-3 (gated, ~800-2000 ms):  run_critic  (20-30% of requests)
  Post   (<1 ms):  derive implied_abnormal

Entry point: await run(article_dict) -> AgentResponse
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from .analyst import run_analyst
from .config import ACTIVE_HORIZONS, HORIZON_DAYS
from .context import build_context
from .critic import apply_critic, run_critic
from .expert import run_expert
from .gate import check_gate, gate_reason
from .schemas import AgentResponse, AnalystOutput, ExplanationBlock, HorizonOutlook
from .sentiment import analyze as run_sentiment
from .sentiment import article_text


def _derive_implied_abnormal(
    ticker_direction: str,
    ticker_magnitude: str,
    sector_direction: str,
    sector_magnitude: str,
) -> str:
    """
    Deterministically derive implied abnormal-return direction from the two legs.

    Rules (in priority order):
    1. ticker up   + sector down/neutral  → up   (outperformance)
    2. ticker down + sector up/neutral    → down (underperformance)
    3. Both same direction:
       - ticker magnitude > sector magnitude → up/down (company-specific alpha)
       - ticker magnitude == sector magnitude → neutral (wash)
    4. Either leg neutral → neutral
    """
    _MAG_RANK = {"none": 0, "small": 1, "medium": 2, "large": 3}

    if ticker_direction == "neutral" or sector_direction == "neutral":
        if ticker_direction == "neutral":
            return "neutral"
        # sector neutral, ticker directional → pure company signal
        return ticker_direction

    if ticker_direction == sector_direction:
        tr = _MAG_RANK.get(ticker_magnitude, 0)
        sr = _MAG_RANK.get(sector_magnitude, 0)
        if tr > sr:
            return ticker_direction   # ticker outpaces sector → abnormal signal
        return "neutral"              # both moving same amount → wash

    # Directions differ → ticker outperforms or underperforms
    return ticker_direction


def _build_response(
    ticker: str,
    event_date: str,
    event_type: str,
    final: AnalystOutput,
    critic_invoked: bool,
    gate_reasons: list[str],
    sentiment_label: str | None = None,
    article_title: str = "",
) -> AgentResponse:
    expl = final.arabic_explanation

    outlook: dict[str, HorizonOutlook] = {}
    for h in ACTIVE_HORIZONS:
        hr = final.horizons[h]
        implied = _derive_implied_abnormal(
            hr.ticker.direction,
            hr.ticker.magnitude,
            hr.sector.direction,
            hr.sector.magnitude,
        )
        outlook[h] = HorizonOutlook(
            horizon_days=HORIZON_DAYS[h],
            ticker=hr.ticker.model_dump(),
            sector=hr.sector.model_dump(),
            implied_abnormal=implied,
            explanation=expl.outlook_by_horizon.get(h, ""),
        )

    # headline falls back to the article title when the model omitted it
    headline = expl.headline or article_title or ""

    explanation_block = ExplanationBlock(
        headline=headline,
        news_story=expl.news_story,
        technical_view=expl.technical_view,
        sentiment_note=expl.sentiment_note,
        outlook_by_horizon=expl.outlook_by_horizon,
        what_could_change_our_view=expl.what_could_change_our_view,
    )

    return AgentResponse(
        status="ok",
        ticker=ticker,
        event_type=event_type,
        processed_at=datetime.now(timezone.utc).isoformat(),
        critic_invoked=critic_invoked,
        outlook=outlook,
        explanation=explanation_block,
        summary=expl.news_story,
        technical_view=expl.technical_view,
        risks=expl.what_could_change_our_view,
        sentiment_label=sentiment_label,
    )


async def run(article: dict) -> AgentResponse:
    """
    Full agent pipeline.

    article must contain:
      ticker       : str   (EGX ticker, e.g. "COMI")
      title        : str   (Arabic headline)
      body         : str   (Arabic full text)
      published_at : str   (ISO date string "YYYY-MM-DD" or full ISO datetime)

    Optional: source, time, datetime
    """
    ticker = article["ticker"]
    # Normalise event_date to YYYY-MM-DD
    raw_date = article.get("published_at") or article.get("date") or article.get("datetime", "")
    event_date = str(raw_date)[:10]

    # ------------------------------------------------------------------
    # Tier-0: build context (deterministic, sequential seed)
    # ------------------------------------------------------------------
    ctx = build_context(ticker, event_date, article)

    # ------------------------------------------------------------------
    # Tier-1: expert + sentiment in parallel
    # ------------------------------------------------------------------
    expert_result, sentiment_result = await asyncio.gather(
        run_expert(ctx, article),
        run_sentiment(article_text(article)),
    )

    # ------------------------------------------------------------------
    # Tier-2: analyst (hot path)
    # ------------------------------------------------------------------
    analyst_output = await run_analyst(ctx, expert_result.model_dump(), sentiment_result, article)

    # ------------------------------------------------------------------
    # Gate: deterministic
    # ------------------------------------------------------------------
    invoke_critic = check_gate(analyst_output, expert_result)
    reasons = gate_reason(analyst_output, expert_result) if invoke_critic else []

    # ------------------------------------------------------------------
    # Tier-3: gated critic
    # ------------------------------------------------------------------
    critic_invoked = False
    final_output = analyst_output
    if invoke_critic:
        critic_output  = await run_critic(ctx, expert_result, sentiment_result, analyst_output, article)
        final_output   = apply_critic(analyst_output, critic_output)
        critic_invoked = True

    return _build_response(
        ticker=ticker,
        event_date=event_date,
        event_type=final_output.event_type,
        final=final_output,
        critic_invoked=critic_invoked,
        gate_reasons=reasons,
        sentiment_label=sentiment_result.label,
        article_title=article.get("title", ""),
    )
