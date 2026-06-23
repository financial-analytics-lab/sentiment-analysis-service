"""
Pydantic schemas for the agent's inputs and outputs.
Every stage of the pipeline uses these types for validation and serialization.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Atoms
# ---------------------------------------------------------------------------

Direction = Literal["up", "down", "neutral"]
Magnitude = Literal["small", "medium", "large", "none"]
Relationship = Literal["company_specific", "sector_moving", "macro"]
DirectionPrior = Literal["positive", "negative", "neutral"]
Strength = Literal["strong", "moderate", "weak", "unclear"]


# ---------------------------------------------------------------------------
# Sentiment
# ---------------------------------------------------------------------------

class SentimentResult(BaseModel):
    label: str
    score: float
    class_probs: dict[str, float]


# ---------------------------------------------------------------------------
# Expert output
# ---------------------------------------------------------------------------

class DirectionalAnalysis(BaseModel):
    direction_prior: DirectionPrior
    strength: Strength
    reasoning: str
    key_factors: list[str] = Field(default_factory=list)


class ExpertOutput(BaseModel):
    event_type: str
    company_analysis: DirectionalAnalysis
    sector_analysis: DirectionalAnalysis
    relationship: Relationship
    risk_flags: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Analyst / Critic per-leg prediction
# ---------------------------------------------------------------------------

class LegPrediction(BaseModel):
    direction: Direction
    magnitude: Magnitude
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class HorizonResult(BaseModel):
    ticker: LegPrediction
    sector: LegPrediction
    implied_abnormal: Optional[Direction] = None


class ArabicExplanation(BaseModel):
    news_story: str
    technical_view: str
    sentiment_note: str
    outlook_by_horizon: dict[str, str]
    what_could_change_our_view: str


class AnalystOutput(BaseModel):
    event_type: str
    ta_signals_cited: list[str] = Field(default_factory=list)
    horizons: dict[str, HorizonResult]
    arabic_explanation: ArabicExplanation


# ---------------------------------------------------------------------------
# Critic output
# ---------------------------------------------------------------------------

class CriticOutput(BaseModel):
    verdict: Literal["confirm", "override"]
    critique: str
    horizons: dict[str, HorizonResult]
    arabic_explanation: Optional[ArabicExplanation] = None


# ---------------------------------------------------------------------------
# API request / response
# ---------------------------------------------------------------------------

class ArticleRequest(BaseModel):
    ticker: str
    title: str
    body: str
    published_at: str
    source: Optional[str] = None


class HorizonOutlook(BaseModel):
    horizon_days: int
    ticker: dict[str, Any]
    sector: dict[str, Any]
    implied_abnormal: Optional[str]


class AgentResponse(BaseModel):
    status: str = "ok"
    ticker: str
    event_type: str
    processed_at: str
    critic_invoked: bool
    outlook: dict[str, HorizonOutlook]
    summary: str
    technical_view: str
    risks: str
