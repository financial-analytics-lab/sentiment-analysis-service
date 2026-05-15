# config.py

# ═══════════════════════════════════════════
#  API KEYS & ENDPOINTS
# ═══════════════════════════════════════════

# Claude (via third-party provider — OpenAI compatible)
CLAUDE_API_KEY = "sk-claudy-8c973ed41ef335f86f68c4560571b00018351d444953a100"
CLAUDE_BASE_URL = "https://api.claudy.cloud"  # ← PUT YOUR PROVIDER URL HERE
CLAUDE_MODEL = "claude-sonnet-4.6"  # ← Ask provider what model names they support

# Groq
GROQ_API_KEY = "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"
GROQ_MODEL = "llama-3.3-70b-versatile"

# Directories
INPUT_DIR = "output/combined-test/"
OUTPUT_DIR_FINBERT = "output/sentiment/finbert/"
OUTPUT_DIR_CLAUDE = "output/sentiment/claude/"
OUTPUT_DIR_GROQ = "output/sentiment/groq/"
ENSEMBLE_DIR = "output/sentiment/ensemble/"

# Rate limiting
CLAUDE_DELAY = 0.5
GROQ_DELAY = 0.5
FINBERT_BATCH_DELAY = 1

# Boilerplate markers
BOILERPLATE_MARKERS = [
    "تعد شركة",
    "تعتبر الشركة من الرواد",
    "يتمثل نشاط الشركة",
]

SOURCE_PREFIXES = [
    "القاهرة - مباشر:",
    "آراب فاينانس:",
]