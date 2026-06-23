"""
Async sentiment analysis using Claude Haiku.

Replaces the CAMeLBERT HuggingFace model with a fast Haiku LLM call.
Returns SentimentResult so no downstream code changes are needed.

Grounding guarantee: the system prompt instructs the model to classify
ONLY based on what is explicitly written in the supplied text.
"""

from __future__ import annotations

from .config import SENTIMENT_MAX_TOKENS, SENTIMENT_MODEL
from .llm_client import _parse_json, complete
from .schemas import SentimentResult

_SENTIMENT_SYSTEM = """\
You are an Arabic financial text sentiment classifier for the Egyptian Stock Exchange (EGX).

TASK: Classify the MARKET SENTIMENT expressed in the given Arabic news article.

DEFINITIONS:
  positive — the text conveys news investors would likely view as good for the stock.
  negative — the text conveys news investors would likely view as bad for the stock.
  neutral  — factual/ambiguous content without a clear directional investor signal.

STRICT GROUNDING RULES:
1. Base your classification ONLY on what is explicitly stated in the text.
2. Do NOT infer future events or prices not mentioned.
3. Do NOT use external knowledge about the company or sector.
4. When the text is ambiguous or mixed, assign higher weight to "neutral".

OUTPUT RULES:
- Output ONLY valid JSON. No markdown fences. No commentary outside the JSON.
- Assign probabilities (0.0–1.0) to each class; they MUST sum to exactly 1.0.

OUTPUT SCHEMA:
{"positive": 0.0, "neutral": 0.0, "negative": 0.0}"""


async def analyze(text: str) -> SentimentResult:
    """Run sentiment classification via Claude Haiku. Returns SentimentResult."""
    if not text or not text.strip():
        return SentimentResult(
            label="neutral",
            score=0.0,
            class_probs={"positive": 0.0, "neutral": 1.0, "negative": 0.0},
        )

    raw = await complete(
        model=SENTIMENT_MODEL,
        messages=[{"role": "user", "content": text}],
        system=_SENTIMENT_SYSTEM,
        max_tokens=SENTIMENT_MAX_TOKENS,
        temperature=0.0,
    )

    try:
        data = _parse_json(raw)
        pos = float(data.get("positive", 0.0))
        neu = float(data.get("neutral",  0.0))
        neg = float(data.get("negative", 0.0))
    except Exception:
        return SentimentResult(
            label="neutral",
            score=0.0,
            class_probs={"positive": 0.0, "neutral": 1.0, "negative": 0.0},
        )

    # Normalize so the three probabilities sum to exactly 1.0
    total = pos + neu + neg
    if total > 0:
        pos, neu, neg = pos / total, neu / total, neg / total
    else:
        pos, neu, neg = 0.0, 1.0, 0.0

    probs = {
        "positive": round(pos, 4),
        "neutral":  round(neu, 4),
        "negative": round(neg, 4),
    }
    label = max(probs, key=probs.get)  # type: ignore[arg-type]
    # Composite score: −1 (strongly negative) → +1 (strongly positive)
    score = round(pos - neg, 4)

    return SentimentResult(label=label, score=score, class_probs=probs)


def article_text(article: dict) -> str:
    """Concatenate headline and body for sentiment input."""
    title = article.get("title", "")
    body  = article.get("body", "")
    return f"{title}\n{body}".strip()
