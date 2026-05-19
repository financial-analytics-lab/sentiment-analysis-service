# arabic_sentiment.py
"""
Arabic financial sentiment analyzer using CAMeLBERT.
Replaces FinBERT for Arabic-language EGX news articles.

Model: CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment
  - Trained on mixed Arabic (MSA + dialectal) sentiment data
  - 3 classes: positive / neutral / negative
  - First run downloads ~500 MB from HuggingFace Hub

Output score: P(positive) - P(negative)  in [-1, +1]
"""

from __future__ import annotations


class ArabicSentimentAnalyzer:

    MODEL_ID = "CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment"

    # Map every possible label variant the model may return to a standard name.
    _LABEL_MAP: dict[str, str] = {
        "positive": "positive", "POSITIVE": "positive",
        "negative": "negative", "NEGATIVE": "negative",
        "neutral":  "neutral",  "NEUTRAL":  "neutral",
        # HuggingFace generic labels (when id2label is not set)
        "LABEL_0":  "negative",
        "LABEL_1":  "neutral",
        "LABEL_2":  "positive",
    }

    _NUMERIC: dict[str, int] = {"positive": 1, "neutral": 0, "negative": -1}

    def __init__(self) -> None:
        print(f"  Loading Arabic sentiment model: {self.MODEL_ID}")
        print("  (first run downloads ~500 MB from HuggingFace Hub)")
        from transformers import pipeline as hf_pipeline  # lazy import
        self._pipe = hf_pipeline(
            "text-classification",
            model=self.MODEL_ID,
            top_k=None,       # return scores for ALL classes (replaces deprecated return_all_scores)
            truncation=True,
            max_length=512,
            device=0,        # CPU; change to 0 if you have a GPU
        )
        print("  Model ready.")

    def analyze(self, text: str) -> dict:
        """
        Analyse one Arabic text string.

        Returns:
            model        : model identifier string
            label        : "positive" / "negative" / "neutral"
            score        : float in [-1, +1]   P(positive) - P(negative)
            numeric      : 1 / 0 / -1
            class_probs  : {"positive": float, "neutral": float, "negative": float}
        """
        if not text or not text.strip():
            return {
                "model": self.MODEL_ID,
                "label": "neutral",
                "score": 0.0,
                "numeric": 0,
                "class_probs": {"positive": 0.0, "neutral": 1.0, "negative": 0.0},
            }

        result = self._pipe(text)

        # Normalise the shape: transformers may return
        #   [[{label, score}, ...]]  (top_k=None, single input)
        #   [{label, score}, ...]    (top_k=None, batched — shouldn't happen here)
        #   {label, score}           (top-1 only — fallback)
        if isinstance(result, list) and len(result) > 0:
            inner = result[0]
        else:
            inner = result

        if isinstance(inner, dict):          # single top-1 prediction dict
            inner = [inner]

        probs: dict[str, float] = {}
        for item in inner:
            mapped = self._LABEL_MAP.get(item["label"], item["label"])
            probs[mapped] = round(item["score"], 4)

        # Composite polarity score: positive probability minus negative probability.
        composite = round(probs.get("positive", 0.0) - probs.get("negative", 0.0), 4)

        # Predicted label = highest probability class.
        pred_label = max(probs, key=probs.get)

        return {
            "model":       self.MODEL_ID,
            "label":       pred_label,
            "score":       composite,
            "numeric":     self._NUMERIC.get(pred_label, 0),
            "class_probs": probs,
        }


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    model = ArabicSentimentAnalyzer()

    samples = [
        ("positive", "أعلنت شركة أبو قير للأسمدة عن توزيع كوبون نقدي قدره 2.30 جنيه لكل سهم."),
        ("negative", "سجلت الشركة خسائر فادحة خلال الربع الثالث بسبب ارتفاع تكاليف الإنتاج."),
        ("neutral",  "عقد مجلس إدارة الشركة اجتماعه الدوري لمناقشة الأداء المالي للعام الماضي."),
    ]

    print("\n" + "=" * 60)
    for expected, text in samples:
        result = model.analyze(text)
        match = "OK" if result["label"] == expected else "MISMATCH"
        print(f"[{match}] expected={expected:<9} got={result['label']:<9} score={result['score']:+.4f}")
        print(f"     {text[:80]}")
        print(f"     probs: {result['class_probs']}\n")
