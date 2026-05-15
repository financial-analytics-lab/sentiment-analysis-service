# groq_analyzer.py
# ═══════════════════════════════════════════
#  Groq (Qwen3-32B): Arabic Chain-of-Thought Sentiment
# ═══════════════════════════════════════════

import json
import time
from datetime import datetime
from tqdm import tqdm
from openai import OpenAI

from config import GROQ_API_KEY, GROQ_MODEL, GROQ_DELAY
from base_analyzer import (
    clean_article_text,
    load_articles,
    save_results,
    build_sentiment_prompt,
)


class GroqAnalyzer:

    LABEL_MAP = {"positive": 1, "neutral": 0, "negative": -1}

    def __init__(self, api_key: str = None):
        key = api_key or GROQ_API_KEY
        self.client = OpenAI(
            api_key=key,
            base_url="https://api.groq.com/openai/v1",
        )
        self.model = GROQ_MODEL
        print(f"✅ Groq analyzer ready (model: {self.model})")

    def analyze_single(self, article: dict) -> dict:
        """Analyze one article with Groq/LLaMA."""
        text = clean_article_text(article)
        prompt = build_sentiment_prompt(text)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=500,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )

            result = json.loads(response.choices[0].message.content)

            article["groq_sentiment"] = result["sentiment"]
            article["groq_confidence"] = result["confidence"]
            article["groq_reasoning_ar"] = result["reasoning_ar"]
            article["groq_reasoning_en"] = result["reasoning_en"]
            article["groq_risk_signals"] = result["risk_signals"]
            article["groq_key_events"] = result["key_financial_events"]
            article["groq_impact_duration"] = result["impact_duration"]
            article["groq_numeric"] = self.LABEL_MAP.get(result["sentiment"], 0)

        except Exception as e:
            print(f"    ⚠️ Groq error: {e}")
            article["groq_sentiment"] = "error"
            article["groq_confidence"] = 0
            article["groq_reasoning_ar"] = ""
            article["groq_reasoning_en"] = str(e)
            article["groq_risk_signals"] = []
            article["groq_key_events"] = []
            article["groq_impact_duration"] = ""
            article["groq_numeric"] = 0

        return article

    def analyze_company_file(self, input_path: str, output_path: str) -> dict:
        """Analyze all articles for one company."""
        data = load_articles(input_path)
        articles = data["articles"]
        print(f"  📰 Analyzing {len(articles)} articles with Groq...")

        enriched = []
        for art in tqdm(articles, desc="  Groq"):
            enriched.append(self.analyze_single(art))
            time.sleep(GROQ_DELAY)

        data["articles"] = enriched
        dist = {
            "positive": sum(1 for a in enriched if a.get("groq_numeric") == 1),
            "neutral": sum(1 for a in enriched if a.get("groq_numeric") == 0),
            "negative": sum(1 for a in enriched if a.get("groq_numeric") == -1),
        }
        data["metadata"]["groq_sentiment"] = {
            "model": f"{self.model} (via Groq)",
            "analyzed_at": datetime.now().isoformat(),
            "total_analyzed": len(enriched),
            "distribution": dist,
        }

        save_results(data, output_path)
        return data


if __name__ == "__main__":
    # Quick test
    analyzer = GroqAnalyzer()
    test = {
        "title": "أبو قير للأسمدة تقرر توزيع كوبون نقدي على المساهمين",
        "body": "أعلنت شركة أبو قير للأسمدة عن توزيع كوبون نقدي قدره 2.30 جنيه لكل سهم",
    }
    result = analyzer.analyze_single(test.copy())
    print(f"\n📰 {result['title']}")
    print(f"🎯 {result['groq_sentiment']} ({result['groq_confidence']})")
    print(f"💬 {result['groq_reasoning_en']}")