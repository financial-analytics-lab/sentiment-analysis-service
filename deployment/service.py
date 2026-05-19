from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from pipeline3 import analyze_article
from prompts import HORIZONS

app = FastAPI(
    title="EGX30 News Impact Service",
    version="1.0.0",
    description="Microservice wrapper around the EGX30 news-impact pipeline.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _extract_article(payload: Any) -> dict:
    if isinstance(payload, dict) and "article" in payload and isinstance(payload["article"], dict):
        return payload["article"]
    if isinstance(payload, dict):
        return payload
    raise HTTPException(status_code=400, detail="Request body must be a JSON object containing one article.")


def _build_user_response(trace: dict) -> dict:
    critic = (trace.get("critic_agent") or {}).get("output") or {}
    final = trace.get("final") or {}

    explanation = (
        (trace.get("arabic_qa") or {}).get("explanation_post")
        or critic.get("explanation_for_user")
        or {}
    )

    per_horizon_pred = final.get("per_horizon") or {}
    outlook_text = explanation.get("outlook_by_horizon") or {}

    outlook: dict = {}
    for h in HORIZONS:
        pred = per_horizon_pred.get(h) or {}
        outlook[h] = {
            "direction":   pred.get("direction"),
            "magnitude":   pred.get("magnitude"),
            "confidence":  pred.get("confidence"),
            "explanation": outlook_text.get(h) or pred.get("reasoning"),
        }

    return {
        "status":         "ok",
        "ticker":         trace.get("ticker"),
        "summary":        explanation.get("news_story"),
        "technical_view": explanation.get("technical_view"),
        "sentiment_note": explanation.get("sentiment_note"),
        "outlook":        outlook,
        "risks":          explanation.get("what_could_change_our_view"),
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(request: Request) -> dict:
    payload = await request.json()
    article = _extract_article(payload)

    try:
        trace = analyze_article(article, save_trace=False)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return _build_user_response(trace)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("service:app", host="0.0.0.0", port=8000, reload=False)
