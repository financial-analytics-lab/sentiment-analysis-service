# run_all.py
# ═══════════════════════════════════════════
#  Main runner — FinBERT + Claude + Groq
# ═══════════════════════════════════════════

import argparse
import os
import sys
import glob

from config import (
    INPUT_DIR, OUTPUT_DIR_FINBERT, OUTPUT_DIR_CLAUDE,
    OUTPUT_DIR_GROQ, ENSEMBLE_DIR,
)
from base_analyzer import get_symbol_from_path


def test_single_article():
    """Quick test with one article."""
    test_article = {
        "title": "أبو قير للأسمدة تقرر توزيع كوبون نقدي على المساهمين",
        "body": "أعلنت شركة أبو قير للأسمدة عن توزيع كوبون نقدي قدره 2.30 جنيه لكل سهم",
    }

    print("\n📰 Test Article:", test_article["title"])
    print(f"{'═'*60}\n")

    # FinBERT
    try:
        from finbert_analyzer import FinBERTAnalyzer
        fb = FinBERTAnalyzer()
        result = fb.analyze_single(test_article.copy())
        print(f"  🔵 FinBERT: {result['sentiment_label']} ({result['sentiment_score']})")
        print(f"     🔤 {result['text_english']}")
    except Exception as e:
        print(f"  🔵 FinBERT: ❌ {str(e)[:80]}")

    print()

    # Claude
    try:
        from claude_analyzer import ClaudeAnalyzer
        cl = ClaudeAnalyzer()
        result = cl.analyze_single(test_article.copy())
        print(f"  🟣 Claude:  {result['claude_sentiment']} ({result['claude_confidence']})")
        print(f"     💬 {result['claude_reasoning_en']}")
    except Exception as e:
        print(f"  🟣 Claude:  ❌ {str(e)[:80]}")

    print()

    # Groq
    try:
        from groq_analyzer import GroqAnalyzer
        gq = GroqAnalyzer()
        result = gq.analyze_single(test_article.copy())
        print(f"  🟢 Groq:    {result['groq_sentiment']} ({result['groq_confidence']})")
        print(f"     💬 {result['groq_reasoning_en']}")
    except Exception as e:
        print(f"  🟢 Groq:    ❌ {str(e)[:80]}")


def run_full_pipeline(use_finbert=True, use_claude=True, use_groq=True):
    """Run sentiment analysis on all company files."""

    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.json")))
    if not files:
        print(f"❌ No JSON files found in {INPUT_DIR}")
        sys.exit(1)

    # ── Determine models to run ──
    models_config = {}
    if use_finbert:
        models_config["finbert"] = {"dir": OUTPUT_DIR_FINBERT, "suffix": "_finbert.json"}
    if use_claude:
        models_config["claude"] = {"dir": OUTPUT_DIR_CLAUDE, "suffix": "_claude.json"}
    if use_groq:
        models_config["groq"] = {"dir": OUTPUT_DIR_GROQ, "suffix": "_groq.json"}

    if not models_config:
        print("❌ No models selected!")
        sys.exit(1)

    print(f"\n{'═'*60}")
    print(f"  📰 EGX Financial News Sentiment Analysis")
    print(f"  Companies:  {len(files)}")
    print(f"  Models:     {', '.join(models_config.keys())}")
    print(f"  Input:      {INPUT_DIR}")
    print(f"{'═'*60}\n")

    # ── Create output dirs ──
    for cfg in models_config.values():
        os.makedirs(cfg["dir"], exist_ok=True)
    os.makedirs(ENSEMBLE_DIR, exist_ok=True)

    # ── Initialize analyzers ──
    analyzers = {}

    if "finbert" in models_config:
        try:
            from finbert_analyzer import FinBERTAnalyzer
            analyzers["finbert"] = FinBERTAnalyzer()
        except Exception as e:
            print(f"⚠️ FinBERT failed to load: {e}")
            print("  Continuing without FinBERT...\n")

    if "claude" in models_config:
        try:
            from claude_analyzer import ClaudeAnalyzer
            analyzers["claude"] = ClaudeAnalyzer()
        except Exception as e:
            print(f"⚠️ Claude failed to load: {e}")
            print("  Continuing without Claude...\n")

    if "groq" in models_config:
        try:
            from groq_analyzer import GroqAnalyzer
            analyzers["groq"] = GroqAnalyzer()
        except Exception as e:
            print(f"⚠️ Groq failed to load: {e}")
            print("  Continuing without Groq...\n")

    if not analyzers:
        print("❌ No models could be loaded!")
        sys.exit(1)

    print(f"\n✅ Active models: {', '.join(analyzers.keys())}")
    print(f"   ({len(analyzers)} out of {len(models_config)} requested)\n")

    # ── Process each company ──
    icons = {"finbert": "🔵", "claude": "🟣", "groq": "🟢"}

    for filepath in files:
        symbol = get_symbol_from_path(filepath)
        print(f"\n{'═'*60}")
        print(f"📈 {symbol}")
        print(f"{'═'*60}")

        for model_name, analyzer in analyzers.items():
            icon = icons.get(model_name, "⚪")
            print(f"\n{icon} {model_name}:")
            output_path = os.path.join(
                models_config[model_name]["dir"],
                f"{symbol}{models_config[model_name]['suffix']}"
            )
            try:
                analyzer.analyze_company_file(filepath, output_path)
            except Exception as e:
                print(f"  ❌ {model_name} failed for {symbol}: {e}")

    print(f"\n{'═'*60}")
    print("✅ ALL MODELS DONE!")
    print(f"{'═'*60}")

    # ── Build ensemble ──
    from ensemble import build_ensemble, print_summary

    print("\n⚖️ Building ensemble...")

    for filepath in files:
        symbol = get_symbol_from_path(filepath)
        ensemble_path = os.path.join(ENSEMBLE_DIR, f"{symbol}_ensemble.json")
        print(f"\n📈 {symbol}")
        result = build_ensemble(symbol, ensemble_path)
        if result["articles"]:
            print(f"  ✅ {len(result['articles'])} articles ensembled")

    # ── Summary ──
    print_summary(ENSEMBLE_DIR)

    print(f"\n📂 Output files:")
    for model_name in analyzers:
        print(f"   {models_config[model_name]['dir']}")
    print(f"   {ENSEMBLE_DIR}")
    print()


def main():
    parser = argparse.ArgumentParser(description="EGX News Sentiment Analysis")
    parser.add_argument("--test", action="store_true", help="Test one article with all models")
    parser.add_argument("--finbert-only", action="store_true", help="FinBERT only")
    parser.add_argument("--claude-only", action="store_true", help="Claude only")
    parser.add_argument("--groq-only", action="store_true", help="Groq only")
    parser.add_argument("--no-finbert", action="store_true", help="Skip FinBERT")
    parser.add_argument("--no-claude", action="store_true", help="Skip Claude")
    parser.add_argument("--no-groq", action="store_true", help="Skip Groq")

    args = parser.parse_args()

    if args.test:
        test_single_article()
    elif args.finbert_only:
        run_full_pipeline(use_finbert=True, use_claude=False, use_groq=False)
    elif args.claude_only:
        run_full_pipeline(use_finbert=False, use_claude=True, use_groq=False)
    elif args.groq_only:
        run_full_pipeline(use_finbert=False, use_claude=False, use_groq=True)
    elif args.no_finbert:
        run_full_pipeline(use_finbert=False, use_claude=True, use_groq=True)
    elif args.no_claude:
        run_full_pipeline(use_finbert=True, use_claude=False, use_groq=True)
    elif args.no_groq:
        run_full_pipeline(use_finbert=True, use_claude=True, use_groq=False)
    else:
        # Default: ALL 3 MODELS
        run_full_pipeline(use_finbert=True, use_claude=True, use_groq=True)


if __name__ == "__main__":
    main()