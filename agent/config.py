"""
Agent-specific configuration.
Reads API credentials from the parent config.py so existing .env / secrets
management works without duplication.
"""

import math
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL  # noqa: E402

# ---------------------------------------------------------------------------
# LLM models
# ---------------------------------------------------------------------------
# Override via environment variables if the proxy supports additional models.
import os

EXPERT_MODEL     = os.getenv("AGENT_EXPERT_MODEL",     "claude-sonnet-4-6")
ANALYST_MODEL    = os.getenv("AGENT_ANALYST_MODEL",    "claude-sonnet-4-6")
CRITIC_MODEL     = os.getenv("AGENT_CRITIC_MODEL",     "claude-sonnet-4-6")
SENTIMENT_MODEL  = os.getenv("AGENT_SENTIMENT_MODEL",  "claude-haiku-4-5-20251001")

EXPERT_MAX_TOKENS    = 512
ANALYST_MAX_TOKENS   = 2048
CRITIC_MAX_TOKENS    = 2048
SENTIMENT_MAX_TOKENS = 128

LLM_TIMEOUT = 90  # seconds per request

# ---------------------------------------------------------------------------
# Prediction horizons (short=5d, medium=10d — large dropped as coin flip)
# ---------------------------------------------------------------------------
ACTIVE_HORIZONS: list[str] = ["short", "medium"]
HORIZON_DAYS: dict[str, int] = {"short": 5, "medium": 10}


def ticker_neutral_band(atr_pct: float | None, horizon: str) -> float:
    n = HORIZON_DAYS[horizon]
    if atr_pct and atr_pct > 0:
        return round(0.5 * atr_pct * math.sqrt(n), 2)
    return {"short": 1.0, "medium": 2.5}[horizon]


def ticker_magnitude_bands(atr_pct: float | None, horizon: str) -> dict:
    n = HORIZON_DAYS[horizon]
    if atr_pct and atr_pct > 0:
        sm = round(0.7 * atr_pct * math.sqrt(n), 2)
        md = round(2.0 * atr_pct * math.sqrt(n), 2)
    else:
        sm, md = {"short": (1.0, 3.0), "medium": (2.5, 6.0)}[horizon]
    return {"small": [0.0, sm], "medium": [sm, md], "large": [md, None]}


def sector_neutral_band(sector_atr: float | None, horizon: str) -> float:
    n = HORIZON_DAYS[horizon]
    if sector_atr and sector_atr > 0:
        return round(0.5 * sector_atr * math.sqrt(n), 2)
    return {"short": 0.8, "medium": 2.0}[horizon]


# ---------------------------------------------------------------------------
# Gate thresholds
# ---------------------------------------------------------------------------
GATE_MIN_CONFIDENCE = float(os.getenv("AGENT_GATE_MIN_CONF", "0.60"))

# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8001"))

# LLM retry
LLM_MAX_RETRIES = 3
LLM_RETRY_DELAY = 1.5  # seconds between retries
