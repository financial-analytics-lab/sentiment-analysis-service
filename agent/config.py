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

# Generous ceilings: the API gateway forces an extended-thinking block, whose
# trace counts toward max_tokens — too low a ceiling truncates the JSON answer
# before a single '{' is emitted. complete_json() also escalates these on a
# parse-failure retry, but we start high to avoid the first miss.
EXPERT_MAX_TOKENS    = 4096
ANALYST_MAX_TOKENS   = 8192
CRITIC_MAX_TOKENS    = 8192
SENTIMENT_MAX_TOKENS = 2048

LLM_TIMEOUT = 90  # seconds per request

# ---------------------------------------------------------------------------
# Prediction horizons (trading-day WINDOW LENGTH, measured from start day):
#   short  = t+0            (immediate reaction, 1 trading day)
#   medium = t+0 .. t+1     (early response, 2 trading days)
#   large  = t+0 .. t+19    (extended impact, one trading month)
# Magnitude / neutral bands rescale automatically by sqrt(N).
# ---------------------------------------------------------------------------
ACTIVE_HORIZONS: list[str] = ["short", "medium", "large"]
HORIZON_DAYS: dict[str, int] = {"short": 1, "medium": 2, "large": 20}

# Human-readable Arabic horizon labels (used in user-facing explanations).
HORIZON_LABEL_AR: dict[str, str] = {
    "short":  "اليوم الأول (t+0): رد الفعل الفوري",
    "medium": "اليومان الأولان (t+0 إلى t+1): الاستجابة المبكرة",
    "large":  "العشرون يوم تداول (t+0 إلى t+19): الأثر الممتد",
}

# Fallback bands (% ) used only when ATR is unavailable. Keyed by horizon.
_FALLBACK_TICKER_NEUTRAL: dict[str, float] = {"short": 1.0, "medium": 1.5, "large": 4.0}
_FALLBACK_TICKER_MAG: dict[str, tuple[float, float]] = {
    "short": (1.0, 3.0), "medium": (1.5, 4.0), "large": (4.0, 10.0),
}
_FALLBACK_SECTOR_NEUTRAL: dict[str, float] = {"short": 0.8, "medium": 1.2, "large": 3.0}


def ticker_neutral_band(atr_pct: float | None, horizon: str) -> float:
    n = HORIZON_DAYS[horizon]
    if atr_pct and atr_pct > 0:
        return round(0.5 * atr_pct * math.sqrt(n), 2)
    return _FALLBACK_TICKER_NEUTRAL[horizon]


def ticker_magnitude_bands(atr_pct: float | None, horizon: str) -> dict:
    n = HORIZON_DAYS[horizon]
    if atr_pct and atr_pct > 0:
        sm = round(0.7 * atr_pct * math.sqrt(n), 2)
        md = round(2.0 * atr_pct * math.sqrt(n), 2)
    else:
        sm, md = _FALLBACK_TICKER_MAG[horizon]
    return {"small": [0.0, sm], "medium": [sm, md], "large": [md, None]}


def sector_neutral_band(sector_atr: float | None, horizon: str) -> float:
    n = HORIZON_DAYS[horizon]
    if sector_atr and sector_atr > 0:
        return round(0.5 * sector_atr * math.sqrt(n), 2)
    return _FALLBACK_SECTOR_NEUTRAL[horizon]


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
