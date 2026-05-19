from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from pipeline3 import analyze_article

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


def _extract_final_summary(trace: dict) -> dict:
    critic = trace.get("critic_agent", {})
    payload = {}
    if isinstance(critic, dict):
        if isinstance(critic.get("parsed"), dict):
            payload = critic["parsed"]
        elif isinstance(critic.get("output"), dict):
            payload = critic["output"]
        else:
            payload = critic

    arabic_qa = trace.get("arabic_qa", {})
    explanation_post = arabic_qa.get("explanation_post") or payload.get("explanation_post", {})

    return {
        "tldr": payload.get("tldr", {}),
        "explanation_for_user": payload.get("explanation_for_user", {}),
        "explanation_post": explanation_post,
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

    summary = _extract_final_summary(trace)

    return {
        "status": "ok",
        "ticker": trace["ticker"],
        "article_date": trace["article_date"],
        "context_date": trace["context_date"],
        "tldr": summary["tldr"],
        "explanation_for_user": summary["explanation_for_user"],
        "explanation_post": summary.get("explanation_post", {}),
        "final": trace.get("final", {}),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("service:app", host="0.0.0.0", port=8000, reload=False)
