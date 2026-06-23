"""
ANALYST agent — Tier-2, dominant cost, hot path.

Receives: dual context + expert output + sentiment result + article.
Produces: per-horizon dual-target predictions (ticker leg + sector leg)
          plus Arabic explanation.
"""

from __future__ import annotations

from .config import ANALYST_MAX_TOKENS, ANALYST_MODEL, ACTIVE_HORIZONS
from .llm_client import _parse_json, complete
from .prompts import ANALYST_SYSTEM, analyst_user
from .schemas import (
    AnalystOutput,
    ArabicExplanation,
    HorizonResult,
    LegPrediction,
    SentimentResult,
)


async def run_analyst(
    ctx: dict,
    expert: dict,
    sentiment: SentimentResult,
    article: dict,
) -> AnalystOutput:
    user_msg = analyst_user(
        ctx,
        expert,
        sentiment.label,
        sentiment.score,
        article,
    )
    raw = await complete(
        model=ANALYST_MODEL,
        messages=[{"role": "user", "content": user_msg}],
        system=ANALYST_SYSTEM,
        max_tokens=ANALYST_MAX_TOKENS,
        temperature=0.0,
    )
    data = _parse_json(raw)
    return _coerce_analyst(data)


def _coerce_leg(block: dict | None, default_direction: str = "neutral") -> LegPrediction:
    if not block:
        return LegPrediction(
            direction=default_direction,  # type: ignore[arg-type]
            magnitude="none",
            confidence=0.5,
            reasoning="",
        )
    direction = block.get("direction", default_direction)
    magnitude = block.get("magnitude", "none")
    # neutral direction must have magnitude "none"
    if direction == "neutral":
        magnitude = "none"
    return LegPrediction(
        direction=direction,        # type: ignore[arg-type]
        magnitude=magnitude,        # type: ignore[arg-type]
        confidence=float(block.get("confidence", 0.5)),
        reasoning=block.get("reasoning", ""),
    )


def _coerce_analyst(data: dict) -> AnalystOutput:
    raw_horizons = data.get("horizons") or {}
    horizons: dict[str, HorizonResult] = {}
    for h in ACTIVE_HORIZONS:
        h_block = raw_horizons.get(h) or {}
        horizons[h] = HorizonResult(
            ticker=_coerce_leg(h_block.get("ticker")),
            sector=_coerce_leg(h_block.get("sector")),
        )

    expl = data.get("arabic_explanation") or {}
    explanation = ArabicExplanation(
        news_story=expl.get("news_story", ""),
        technical_view=expl.get("technical_view", ""),
        sentiment_note=expl.get("sentiment_note", ""),
        outlook_by_horizon=expl.get("outlook_by_horizon") or {},
        what_could_change_our_view=expl.get("what_could_change_our_view", ""),
    )

    return AnalystOutput(
        event_type=data.get("event_type", "other"),
        ta_signals_cited=data.get("ta_signals_cited") or [],
        horizons=horizons,
        arabic_explanation=explanation,
    )
