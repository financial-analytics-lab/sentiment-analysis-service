# base_analyzer.py
# ═══════════════════════════════════════════
#  Shared utilities for all analyzers
# ═══════════════════════════════════════════

import json
import os
from config import BOILERPLATE_MARKERS, SOURCE_PREFIXES


def clean_article_text(article: dict, max_length: int = 600) -> str:
    """
    Extract and clean text from an article.
    Removes source prefixes and boilerplate company descriptions.
    """
    title = article.get("title", "").strip()
    body = article.get("body", "").strip()

    # Remove source prefixes
    for prefix in SOURCE_PREFIXES:
        body = body.replace(prefix, "").strip()

    # Remove boilerplate company descriptions
    for marker in BOILERPLATE_MARKERS:
        if marker in body:
            body = body[:body.index(marker)].strip()

    if body:
        return f"{title}\n{body[:max_length]}"
    return title


def load_articles(input_path: str) -> dict:
    """Load articles from a JSON file."""
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_results(data: dict, output_path: str):
    """Save results to a JSON file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  💾 Saved: {output_path}")


def get_symbol_from_path(filepath: str) -> str:
    """Extract company symbol from file path."""
    return os.path.basename(filepath).replace("_combined", "").replace(".json", "")


def build_sentiment_prompt(text: str) -> str:
    """Build the sentiment analysis prompt for API models."""
    return f"""You are a financial analyst specializing in the Egyptian Stock Exchange (EGX).
Analyze the sentiment of this Arabic financial news article and explain your reasoning.

Article:
{text}

Respond ONLY with valid JSON in this exact format, no other text:
{{
    "sentiment": "positive" or "negative" or "neutral",
    "confidence": 0.0 to 1.0,
    "reasoning_ar": "شرح بالعربي في جملتين لماذا هذا الخبر إيجابي أو سلبي أو محايد",
    "reasoning_en": "Explain in 2-3 sentences WHY this sentiment was assigned",
    "risk_signals": ["list of risk factors if any, empty list if none"],
    "key_financial_events": ["e.g. dividend, earnings_growth, expansion, debt, loss"],
    "impact_duration": "short-term" or "medium-term" or "long-term"
}}"""