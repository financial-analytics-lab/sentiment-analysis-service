# finbert_analyzer.py
# ═══════════════════════════════════════════
#  FinBERT: Translate Arabic → English → Financial BERT
# ═══════════════════════════════════════════

import time
import torch
from datetime import datetime
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from deep_translator import GoogleTranslator

from config import FINBERT_BATCH_DELAY
from base_analyzer import clean_article_text, load_articles, save_results


class FinBERTAnalyzer:

    LABEL_MAP = {"positive": 1, "neutral": 0, "negative": -1}

    def __init__(self):
        print("🔄 Loading FinBERT...")
        model_name = "ProsusAI/finbert"

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Handle PyTorch CVE-2025-32434
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, weights_only=True
            )
        except TypeError:
            model = AutoModelForSequenceClassification.from_pretrained(model_name)

        self.pipe = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            truncation=True,
            max_length=512,
        )
        self.translator = GoogleTranslator(source="ar", target="en")
        print("✅ FinBERT + Translator ready")

    def _translate(self, text: str) -> str:
        """Translate Arabic text to English."""
        try:
            return self.translator.translate(text[:500])
        except Exception as e:
            print(f"    ⚠️ Translation error: {e}")
            return text

    def analyze_single(self, article: dict) -> dict:
        """Analyze one article."""
        arabic_text = clean_article_text(article, max_length=500)
        english_text = self._translate(arabic_text)
        result = self.pipe(english_text)[0]

        article["text_arabic"] = arabic_text[:200]
        article["text_english"] = english_text[:200]
        article["sentiment_label"] = result["label"]
        article["sentiment_score"] = round(result["score"], 4)
        article["sentiment_numeric"] = self.LABEL_MAP.get(result["label"], 0)

        return article

    def analyze_company_file(self, input_path: str, output_path: str) -> dict:
        """Analyze all articles for one company."""
        data = load_articles(input_path)
        articles = data["articles"]
        print(f"  📰 Translating & analyzing {len(articles)} articles...")

        enriched = []
        for i, art in enumerate(tqdm(articles, desc="  FinBERT")):
            enriched.append(self.analyze_single(art))
            if i % 10 == 0 and i > 0:
                time.sleep(FINBERT_BATCH_DELAY)

        data["articles"] = enriched
        data["metadata"]["finbert_sentiment"] = {
            "model": "ProsusAI/finbert (with Arabic→English translation)",
            "analyzed_at": datetime.now().isoformat(),
            "total_analyzed": len(enriched),
            "distribution": {
                "positive": sum(1 for a in enriched if a["sentiment_numeric"] == 1),
                "neutral": sum(1 for a in enriched if a["sentiment_numeric"] == 0),
                "negative": sum(1 for a in enriched if a["sentiment_numeric"] == -1),
            },
        }

        save_results(data, output_path)
        return data


if __name__ == "__main__":
    # Quick test
    analyzer = FinBERTAnalyzer()
    test = {
        "title": "أبو قير للأسمدة تقرر توزيع كوبون نقدي على المساهمين",
        "body": "أعلنت شركة أبو قير للأسمدة عن توزيع كوبون نقدي قدره 2.30 جنيه لكل سهم",
    }
    result = analyzer.analyze_single(test.copy())
    print(f"\n📰 {result['title']}")
    print(f"🔤 {result['text_english']}")
    print(f"🎯 {result['sentiment_label']} ({result['sentiment_score']})")