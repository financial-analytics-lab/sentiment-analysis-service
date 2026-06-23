"""
EGX News-Impact Agent — dual-target LLM-only pipeline.

Predicts:
  • ticker absolute return direction (up / down / neutral)
  • sector index return direction (up / down / neutral)

over short (5-day) and medium (10-day) horizons, plus a derived
implied_abnormal direction, from a single Arabic news article.

Entry points:
  from agent.pipeline import run          # async, main orchestrator
  from agent.service import app           # FastAPI app (port 8001)
"""

from .pipeline import run

__all__ = ["run"]
