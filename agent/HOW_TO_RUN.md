# How to Run the EGX News-Impact Agent (v2)

The agent lives entirely in `agent/`. It exposes a FastAPI service on **port 8001**
and is completely separate from the legacy pipeline on port 8000.

---

## 1. Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.11 or higher |
| pip | any recent version |
| Price data | `egyptian_stocks_2020_2025.json` in the repo root |

**You do NOT need a GPU.** All inference is via LLM API calls. No local model downloads.

---

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

The agent uses only packages already in `requirements.txt`:
`fastapi`, `uvicorn`, `requests`, `pydantic`, `pandas`, `numpy`, `python-dotenv`.

> The HuggingFace `transformers` and `torch` packages listed in requirements.txt
> are still needed by the legacy pipeline but are **not used** by the v2 agent.

---

## 3. Configure environment variables

Copy the example file and fill in your values:

```bash
cp .env.example .env
```

Open `.env` and set:

```ini
# Required — Anthropic gateway credentials
ANTHROPIC_API_KEY="sk-claudy-d84413544c9539152484be0de5ac8a43de30879fe7cb33f3"
ANTHROPIC_BASE_URL=https://api.claudy.cloud   # or https://api.anthropic.com

# Optional model overrides (defaults shown)
AGENT_EXPERT_MODEL=claude-sonnet-4-6
AGENT_ANALYST_MODEL=claude-sonnet-4-6
AGENT_CRITIC_MODEL=claude-sonnet-4-6
AGENT_SENTIMENT_MODEL=claude-haiku-4-5

# Optional service config
API_HOST=0.0.0.0
API_PORT=8001
AGENT_GATE_MIN_CONF=0.60
```

> **Security:** never commit `.env`. The `.gitignore` already excludes it.

---

## 4. Seed price data (one-time)

The agent reads price data through `features.py`. The raw JSON file is already
present in the repo. Run the data-preparation script once to generate the parquet
files used by the evaluator:

```bash
python prepare_data.py
```

This creates:
- `returns_daily.parquet` — daily log returns per ticker
- `sector_returns_daily.parquet` — sector-level returns including `{sector}_excl_{ticker}` columns

> If these parquet files already exist in the repo, you can skip this step.

---

## 5. Run the agent service

```bash
python -m agent.service
```

Or with uvicorn directly for more control:

```bash
uvicorn agent.service:app --host 0.0.0.0 --port 8001 --reload
```

The service starts at `http://localhost:8001`.

Verify it is running:

```bash
curl http://localhost:8001/health
# {"status":"ok","version":"2.0.0"}
```

---

## 6. Send a sample prediction request

```bash
curl -s -X POST http://localhost:8001/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "COMI",
    "title": "البنك التجاري الدولي يعلن توزيع أرباح نقدية بواقع 3 جنيهات للسهم",
    "body": "أعلن البنك التجاري الدولي-مصر (كومي) اليوم عن توزيع أرباح نقدية بواقع 3 جنيهات للسهم عن العام المالي الماضي، وذلك بعد موافقة الجمعية العمومية للمساهمين.",
    "published_at": "2025-03-15"
  }' | python -m json.tool
```

---

## 7. Docker (optional)

```bash
docker compose up --build
```

The `docker-compose.yml` maps port 8001. Make sure your `.env` file is present
before building — it is injected at runtime via `env_file`.

---

## 8. Run the evaluator

After collecting prediction JSON files (one per article, saved by your batch script),
run the dual-target evaluator:

```bash
python -m agent.evaluate \
  --predictions-dir agent/outputs/predictions \
  --returns         returns_daily.parquet \
  --sector-returns  sector_returns_daily.parquet \
  --out             agent/outputs/evaluation
```

**Prediction file format** (one JSON per article, saved by your ingestion script):

```json
{
  "ticker": "COMI",
  "event_date": "2025-03-15",
  "in_trading_hours": true,
  "event_type": "dividend",
  "sentiment_label": "positive",
  "context": {
    "ticker_atr": 1.42,
    "sector_atr": 0.91
  },
  "outlook": {
    "short": {
      "ticker":  {"direction": "up",      "confidence": 0.72},
      "sector":  {"direction": "neutral", "confidence": 0.63},
      "implied_abnormal": "up"
    },
    "medium": {
      "ticker":  {"direction": "up",      "confidence": 0.65},
      "sector":  {"direction": "up",      "confidence": 0.58},
      "implied_abnormal": "neutral"
    }
  }
}
```

Outputs written to `agent/outputs/evaluation/`:
- `scorecard.csv` — one row per (article × horizon) with all hit flags and baselines
- `aggregate.json` — overall accuracy, per-horizon breakdown, baseline comparisons

---

## 9. Interactive API docs

While the service is running, open in your browser:

| URL | Description |
|---|---|
| `http://localhost:8001/docs` | Swagger UI — try the API interactively |
| `http://localhost:8001/redoc` | ReDoc — clean read-only reference |

---

## 10. Pipeline architecture (quick reference)

```
Article
  │
  ▼
build_context (~15 ms)          ← Tier-0: deterministic TA context
  │
  ├──► Financial Expert LLM     ─┐
  │    (event interpretation)    │  Tier-1: parallel (~200–400 ms)
  └──► Sentiment (Haiku)        ─┘
  │
  ▼
Analyst LLM                     ← Tier-2: dual-target predictions (~400–1200 ms)
  │
[Gate: low-confidence / contradiction?]
  │ yes
  ▼
Critic LLM (gated, ~20-30%)     ← Tier-3: audit + optional override
  │
  ▼
JSON response
```

Two LLM calls on the critical path (Expert + Analyst). Critic fires only when
the gate triggers. Sentiment runs in parallel with the Expert and adds zero
wall-clock time to the critical path.
