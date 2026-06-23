"""
GATE — deterministic decision node.

Decides whether to invoke the Critic based on the analyst output and the
expert's relationship classification. Pure Python, <1 ms, no LLM call.
"""

from __future__ import annotations

from .config import ACTIVE_HORIZONS, GATE_MIN_CONFIDENCE
from .schemas import AnalystOutput, ExpertOutput


def check_gate(analyst: AnalystOutput, expert: ExpertOutput) -> bool:
    """
    Return True (invoke Critic) when ANY of the following hold:

    G1. Any leg on any horizon has confidence < GATE_MIN_CONFIDENCE.
    G2. Analyst predicts a STRONG sector move (not neutral) on a
        company_specific event without independent sector TA signal
        (the system prompt instructs the analyst to default neutral,
        but we verify in case the model drifted).
    G3. Ticker leg disagrees directionally between short and medium
        (contradictory outlook horizon-to-horizon suggests high uncertainty).
    """
    # G1 — low confidence on any leg
    for h in ACTIVE_HORIZONS:
        hr = analyst.horizons.get(h)
        if hr is None:
            return True  # missing horizon → always escalate
        if hr.ticker.confidence < GATE_MIN_CONFIDENCE:
            return True
        if hr.sector.confidence < GATE_MIN_CONFIDENCE:
            return True

    # G2 — analyst claims significant sector move on company_specific event
    if expert.relationship == "company_specific":
        for h in ACTIVE_HORIZONS:
            hr = analyst.horizons[h]
            if hr.sector.direction != "neutral" and hr.sector.confidence >= 0.70:
                return True  # unexplained sector signal → review

    # G3 — ticker direction flips between short and medium
    if len(ACTIVE_HORIZONS) >= 2:
        h0, h1 = ACTIVE_HORIZONS[0], ACTIVE_HORIZONS[1]
        hr0 = analyst.horizons.get(h0)
        hr1 = analyst.horizons.get(h1)
        if hr0 and hr1:
            d0 = hr0.ticker.direction
            d1 = hr1.ticker.direction
            # Flipping from up→down or down→up (not neutral) is a red flag
            if d0 != "neutral" and d1 != "neutral" and d0 != d1:
                return True

    return False


def gate_reason(analyst: AnalystOutput, expert: ExpertOutput) -> list[str]:
    """Return human-readable list of triggered gate conditions (for tracing)."""
    reasons: list[str] = []

    for h in ACTIVE_HORIZONS:
        hr = analyst.horizons.get(h)
        if hr is None:
            reasons.append(f"G1: missing horizon {h}")
            continue
        if hr.ticker.confidence < GATE_MIN_CONFIDENCE:
            reasons.append(f"G1: ticker {h} confidence {hr.ticker.confidence:.2f} < {GATE_MIN_CONFIDENCE}")
        if hr.sector.confidence < GATE_MIN_CONFIDENCE:
            reasons.append(f"G1: sector {h} confidence {hr.sector.confidence:.2f} < {GATE_MIN_CONFIDENCE}")

    if expert.relationship == "company_specific":
        for h in ACTIVE_HORIZONS:
            hr = analyst.horizons.get(h)
            if hr and hr.sector.direction != "neutral" and hr.sector.confidence >= 0.70:
                reasons.append(
                    f"G2: company_specific event but analyst predicts "
                    f"sector {hr.sector.direction} ({h}, conf={hr.sector.confidence:.2f})"
                )

    if len(ACTIVE_HORIZONS) >= 2:
        h0, h1 = ACTIVE_HORIZONS[0], ACTIVE_HORIZONS[1]
        hr0 = analyst.horizons.get(h0)
        hr1 = analyst.horizons.get(h1)
        if hr0 and hr1:
            d0, d1 = hr0.ticker.direction, hr1.ticker.direction
            if d0 != "neutral" and d1 != "neutral" and d0 != d1:
                reasons.append(f"G3: ticker direction flip {d0}({h0}) → {d1}({h1})")

    return reasons
