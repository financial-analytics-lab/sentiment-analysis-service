"""
FastAPI service wrapper for the EGX News-Impact Agent.
Runs on port 8001 (separate from the legacy service on 8000).

POST /analyze   — main prediction endpoint
GET  /health    — liveness check
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from .config import API_HOST, API_PORT
from .pipeline import run
from .schemas import AgentResponse, ArticleRequest

app = FastAPI(
    title="EGX News-Impact Agent (v2)",
    version="2.0.0",
    description=(
        "LLM-only agentic pipeline: dual-target prediction of ticker absolute return "
        "direction AND sector return direction from a single Arabic news article."
    ),
)

# Explicit origins should be pinned in production.
# allow_credentials is intentionally omitted to avoid the CORS
# allow_origins=* + allow_credentials=True invalid combination (S4 fix).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type", "Authorization"],
)


def _extract_article(payload: dict) -> dict:
    """
    Accept two input shapes:
      {"ticker": "...", "title": "...", "body": "...", "published_at": "..."}
      {"article": {"ticker": ..., ...}}
    """
    if "article" in payload and isinstance(payload["article"], dict):
        inner = dict(payload["article"])
        # top-level ticker overrides if supplied
        if "ticker" in payload:
            inner["ticker"] = payload["ticker"]
        return inner
    return payload


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": "2.0.0"}


@app.post("/analyze", response_model=AgentResponse)
async def analyze(request: Request) -> AgentResponse:
    try:
        payload: dict = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Request body must be valid JSON.")

    article = _extract_article(payload)

    required = {"ticker", "title", "body", "published_at"}
    missing = required - article.keys()
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required fields: {sorted(missing)}",
        )

    try:
        result = await run(article)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("agent.service:app", host=API_HOST, port=API_PORT, reload=False)
