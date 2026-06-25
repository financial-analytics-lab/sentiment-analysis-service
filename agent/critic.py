"""
CRITIC agent — Tier-3, gated, fires on ~20-30% of requests.

Receives: full analyst output + all original context.
Can confirm or override either leg on any horizon.
Uses a stronger model (CRITIC_MODEL) to minimise false overrides.
"""

from __future__ import annotations

from .analyst import _coerce_analyst
from .config import CRITIC_MAX_TOKENS, CRITIC_MODEL
from .llm_client import complete_json
from .prompts import CRITIC_SYSTEM, critic_user
from .schemas import AnalystOutput, CriticOutput, ExpertOutput, SentimentResult


async def run_critic(
    ctx: dict,
    expert: ExpertOutput,
    sentiment: SentimentResult,
    analyst: AnalystOutput,
    article: dict,
) -> CriticOutput:
    """Call the Critic LLM. Returns CriticOutput."""
    user_msg = critic_user(
        ctx,
        expert.model_dump(),
        sentiment.label,
        sentiment.score,
        analyst.model_dump(),
        article,
    )
    data = await complete_json(
        model=CRITIC_MODEL,
        messages=[{"role": "user", "content": user_msg}],
        system=CRITIC_SYSTEM,
        max_tokens=CRITIC_MAX_TOKENS,
        temperature=0.0,
    )
    return _coerce_critic(data)


def _coerce_critic(data: dict) -> CriticOutput:
    verdict   = data.get("verdict", "confirm")
    critique  = data.get("critique", "")
    # Reuse analyst coercion for the horizons block
    analyst_like = {
        "event_type":      data.get("event_type", "other"),
        "ta_signals_cited": data.get("ta_signals_cited", []),
        "horizons":        data.get("horizons", {}),
        "arabic_explanation": data.get("arabic_explanation", {}),
    }
    analyst_coerced = _coerce_analyst(analyst_like)
    return CriticOutput(
        verdict=verdict,    # type: ignore[arg-type]
        critique=critique,
        horizons=analyst_coerced.horizons,
        arabic_explanation=analyst_coerced.arabic_explanation,
    )


def apply_critic(analyst: AnalystOutput, critic: CriticOutput) -> AnalystOutput:
    """
    Merge critic output into the analyst output.
    If verdict="confirm", returns analyst unchanged.
    If verdict="override", replaces horizons (and optionally arabic_explanation).
    """
    if critic.verdict == "confirm":
        return analyst
    return AnalystOutput(
        event_type=analyst.event_type,
        ta_signals_cited=analyst.ta_signals_cited,
        horizons=critic.horizons,
        arabic_explanation=(
            critic.arabic_explanation
            if critic.arabic_explanation is not None
            else analyst.arabic_explanation
        ),
    )
