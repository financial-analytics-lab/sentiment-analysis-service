"""
FINANCIAL EXPERT agent — Tier-1 parallel, fast LLM.

Receives: article + dual context (ticker + sector).
Produces: structured event interpretation (company_analysis, sector_analysis,
          relationship, risk_flags). Does NOT predict direction.
"""

from __future__ import annotations

from .config import EXPERT_MAX_TOKENS, EXPERT_MODEL
from .llm_client import complete_json
from .prompts import EXPERT_SYSTEM, expert_user
from .schemas import ExpertOutput


async def run_expert(ctx: dict, article: dict) -> ExpertOutput:
    """
    Call the Financial Expert LLM.
    Returns a validated ExpertOutput; raises on LLM or parse failure.
    """
    user_msg = expert_user(ctx, article)
    data = await complete_json(
        model=EXPERT_MODEL,
        messages=[{"role": "user", "content": user_msg}],
        system=EXPERT_SYSTEM,
        max_tokens=EXPERT_MAX_TOKENS,
        temperature=0.0,
    )
    return _coerce_expert(data)


def _coerce_expert(data: dict) -> ExpertOutput:
    """
    Coerce the raw LLM JSON into ExpertOutput.
    Fills missing optional fields with safe defaults so a partial response
    doesn't abort the pipeline.
    """
    def _analysis(block: dict, default_prior: str = "neutral") -> dict:
        return {
            "direction_prior": block.get("direction_prior", default_prior),
            "strength":        block.get("strength", "unclear"),
            "reasoning":       block.get("reasoning", ""),
            "key_factors":     block.get("key_factors") or [],
        }

    company = data.get("company_analysis") or {}
    sector  = data.get("sector_analysis") or {}

    return ExpertOutput(
        event_type=data.get("event_type", "other"),
        company_analysis=_analysis(company, "neutral"),
        sector_analysis=_analysis(sector, "neutral"),
        relationship=data.get("relationship", "company_specific"),
        risk_flags=data.get("risk_flags") or [],
    )
