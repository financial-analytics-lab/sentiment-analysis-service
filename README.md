# sentiment-analysis-service
Sentiment Analysis for news

## Microservice endpoint

Run the API with:

```bash
uvicorn service:app --host 0.0.0.0 --port 8000
```

Environment
-----------

1. Copy `.env.example` to `.env` and fill in your API keys.

```bash
cp .env.example .env
# edit .env and add CLAUDE_API_KEY and GROQ_API_KEY
```

Run locally (install the requirements first):

```bash
pip install -r requirements.txt
uvicorn service:app --host 0.0.0.0 --port 8000
```

Docker
------

Build and run with Docker (requires Docker installed):

```bash
docker build -t sentiment-service .
docker run --env-file .env -p 8000:8000 sentiment-service
```

Or with docker-compose:

```bash
docker-compose up --build
```


POST one article to:

```http
POST /analyze
```

Request body can be either the article object itself or wrapped as `{ "article": { ... } }`.

Example payload:

```json
{
	"symbol": "EMFD",
	"company": "Emaar Misr for Development",
	"isin": "EGS673Y1C015",
	"sector": "Real Estate",
	"id": "",
	"date": "2025-01-02",
	"time": "10:54:00",
	"datetime": "2025-01-02 10:54:00",
	"title": "...",
	"body": "...",
	"teaser": "",
	"section": "",
	"source": "Arab Finance",
	"categories": "",
	"country": "",
	"reads": 0,
	"url": "...",
	"image_url": "...",
	"date_raw": "...",
	"_source": "arabfinance"
}
```

Health check:

```http
GET /health
```
