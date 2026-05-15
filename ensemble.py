# ensemble.py
# ═══════════════════════════════════════════
#  Ensemble: Combine FinBERT + Claude + Groq
#  Works with any combination (1, 2, or 3)
# ═══════════════════════════════════════════

import json
import os
import glob

from config import (
    OUTPUT_DIR_FINBERT, OUTPUT_DIR_CLAUDE, OUTPUT_DIR_GROQ,
    ENSEMBLE_DIR, INPUT_DIR,
)
from base_analyzer import save_results, get_symbol_from_path


# ═══════════════════════════════════════════
#  Field mapping for each model
# ═══════════════════════════════════════════
MODEL_FIELDS = {
    "finbert": {
        "sentiment": "sentiment_label",
        "confidence": "sentiment_score",
        "numeric": "sentiment_numeric",
        "reasoning_ar": None,
        "reasoning_en": None,
        "risk_signals": None,
        "key_events": None,
        "impact_duration": None,
    },
    "claude": {
        "sentiment": "claude_sentiment",
        "confidence": "claude_confidence",
        "numeric": "claude_numeric",
        "reasoning_ar": "claude_reasoning_ar",
        "reasoning_en": "claude_reasoning_en",
        "risk_signals": "claude_risk_signals",
        "key_events": "claude_key_events",
        "impact_duration": "claude_impact_duration",
    },
    "groq": {
        "sentiment": "groq_sentiment",
        "confidence": "groq_confidence",
        "numeric": "groq_numeric",
        "reasoning_ar": "groq_reasoning_ar",
        "reasoning_en": "groq_reasoning_en",
        "risk_signals": "groq_risk_signals",
        "key_events": "groq_key_events",
        "impact_duration": "groq_impact_duration",
    },
}

MODEL_DIRS = {
    "finbert": OUTPUT_DIR_FINBERT,
    "claude": OUTPUT_DIR_CLAUDE,
    "groq": OUTPUT_DIR_GROQ,
}

MODEL_SUFFIXES = {
    "finbert": "_finbert.json",
    "claude": "_claude.json",
    "groq": "_groq.json",
}


def _extract_model_output(article: dict, model_name: str) -> dict:
    """Extract standardized output from a model's article data."""
    fields = MODEL_FIELDS[model_name]

    result = {
        "label": article.get(fields["sentiment"], "neutral"),
        "confidence": article.get(fields["confidence"], 0),
    }

    for extra in ["reasoning_ar", "reasoning_en", "risk_signals", "key_events", "impact_duration"]:
        if fields[extra]:
            default = "" if "reasoning" in extra or "duration" in extra else []
            result[extra] = article.get(fields[extra], default)

    return result


def _get_numeric(article: dict, model_name: str) -> int:
    """Get numeric sentiment value from article."""
    field = MODEL_FIELDS[model_name]["numeric"]
    return article.get(field, 0)


def _calculate_weights(available_models: list) -> dict:
    """Calculate normalized weights based on available models."""
    base_weights = {
        "finbert": 0.20,
        "claude": 0.40,
        "groq": 0.40,
    }

    raw = {m: base_weights[m] for m in available_models}
    total = sum(raw.values())
    return {m: round(w / total, 4) for m, w in raw.items()}


def detect_available_models(symbol: str) -> list:
    """Check which model output files exist for a given symbol."""
    available = []
    for model_name, directory in MODEL_DIRS.items():
        path = os.path.join(directory, f"{symbol}{MODEL_SUFFIXES[model_name]}")
        if os.path.exists(path):
            available.append(model_name)
    return available


def build_ensemble(symbol: str, output_path: str) -> dict:
    """Build ensemble from whatever models are available."""
    available = detect_available_models(symbol)

    if not available:
        print(f"  ❌ No model outputs found for {symbol}")
        return {"metadata": {}, "articles": []}

    print(f"  📦 Models found: {', '.join(available)}")

    # Load all available model data
    model_data = {}
    for model_name in available:
        path = os.path.join(MODEL_DIRS[model_name], f"{symbol}{MODEL_SUFFIXES[model_name]}")
        with open(path, "r", encoding="utf-8") as f:
            model_data[model_name] = json.load(f)

    first_model = available[0]
    metadata = model_data[first_model]["metadata"]

    weights = _calculate_weights(available)
    print(f"  ⚖️  Weights: {weights}")

    # Get article count
    article_counts = {m: len(model_data[m]["articles"]) for m in available}
    min_count = min(article_counts.values())

    if len(set(article_counts.values())) > 1:
        print(f"  ⚠️  Article count mismatch: {article_counts} — using first {min_count}")

    # Build ensemble
    ensemble_articles = []

    for i in range(min_count):
        ensemble_score = 0
        labels = []

        for model_name in available:
            art = model_data[model_name]["articles"][i]
            numeric = _get_numeric(art, model_name)
            ensemble_score += numeric * weights[model_name]

            field = MODEL_FIELDS[model_name]["sentiment"]
            labels.append(art.get(field, "neutral"))

        # Determine ensemble label
        if ensemble_score > 0.2:
            ensemble_label = "positive"
        elif ensemble_score < -0.2:
            ensemble_label = "negative"
        else:
            ensemble_label = "neutral"

        # Calculate agreement
        if len(available) > 1:
            agreement = max(labels.count(l) for l in set(labels)) / len(available)
        else:
            agreement = 1.0

        # Build article
        first_art = model_data[first_model]["articles"][i]
        article = {
            "date": first_art.get("date"),
            "title": first_art.get("title"),
        }

        for model_name in available:
            art = model_data[model_name]["articles"][i]
            article[model_name] = _extract_model_output(art, model_name)

        article["ensemble"] = {
            "label": ensemble_label,
            "score": round(ensemble_score, 4),
            "models_used": available,
            "weights": weights,
            "model_agreement": round(agreement, 2),
            "all_agree": agreement == 1.0,
            "model_labels": {m: labels[j] for j, m in enumerate(available)},
        }

        ensemble_articles.append(article)

    output = {
        "metadata": {
            **metadata,
            "ensemble_info": {
                "models_used": available,
                "weights": weights,
                "total_articles": len(ensemble_articles),
            },
        },
        "articles": ensemble_articles,
    }

    save_results(output, output_path)
    return output


def print_summary(ensemble_dir: str):
    """Print summary that adapts to whatever models were used."""
    print(f"\n{'═'*60}")
    print("📋 FINAL SUMMARY")
    print(f"{'═'*60}\n")

    for filepath in sorted(glob.glob(os.path.join(ensemble_dir, "*.json"))):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        symbol = os.path.basename(filepath).replace("_ensemble.json", "")
        articles = data["articles"]

        if not articles:
            continue

        models_used = articles[0].get("ensemble", {}).get("models_used", [])

        print(f"📈 {symbol} ({len(articles)} articles)")
        print(f"   Models: {', '.join(models_used)}")
        print(f"{'─'*50}")

        for model in models_used:
            labels = []
            for a in articles:
                if model in a:
                    labels.append(a[model].get("label", "neutral"))
            if labels:
                pos = labels.count("positive")
                neu = labels.count("neutral")
                neg = labels.count("negative")
                err = labels.count("error")
                line = f"  {model:<10} → ✅ +{pos}  ⚪ ={neu}  ❌ -{neg}"
                if err:
                    line += f"  ⚠️ err={err}"
                print(line)

        ens_labels = [a["ensemble"]["label"] for a in articles]
        pos = ens_labels.count("positive")
        neu = ens_labels.count("neutral")
        neg = ens_labels.count("negative")
        print(f"  {'ensemble':<10} → ✅ +{pos}  ⚪ ={neu}  ❌ -{neg}")

        agree_count = sum(1 for a in articles if a["ensemble"].get("all_agree", False))
        pct = agree_count / len(articles) * 100 if articles else 0
        print(f"  Agreement → {agree_count}/{len(articles)} ({pct:.0f}%)")
        print()


if __name__ == "__main__":
    # Run ensemble for all companies
    os.makedirs(ENSEMBLE_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.json")))

    for filepath in files:
        symbol = get_symbol_from_path(filepath)
        ensemble_path = os.path.join(ENSEMBLE_DIR, f"{symbol}_ensemble.json")
        print(f"\n📈 {symbol}")
        result = build_ensemble(symbol, ensemble_path)
        if result["articles"]:
            print(f"  ✅ {len(result['articles'])} articles ensembled")

    print_summary(ENSEMBLE_DIR)