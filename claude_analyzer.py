# claude_analyzer.py
# ═══════════════════════════════════════════
#  Claude via Anthropic-Native API Gateway
#  Uses /v1/messages with x-api-key auth
# ═══════════════════════════════════════════

import json
import time
import requests
from datetime import datetime
from tqdm import tqdm

from config import CLAUDE_API_KEY, CLAUDE_MODEL, CLAUDE_DELAY
from base_analyzer import (
    clean_article_text,
    load_articles,
    save_results,
    build_sentiment_prompt,
)


class ClaudeAnalyzer:

    LABEL_MAP = {"positive": 1, "neutral": 0, "negative": -1}

    def __init__(self, api_key: str = None, base_url: str = None):
        self.key = api_key or CLAUDE_API_KEY
        self.model = CLAUDE_MODEL
        self.base_url = (base_url or "https://api.claudy.cloud").rstrip("/")
        self.debug = True  # Set False after first success
        print(f"✅ Claude analyzer ready (Anthropic-native)")
        print(f"   Endpoint: {self.base_url}/v1/messages")
        print(f"   Model:    {self.model}")

    # ─────────────────────────────────────
    #  Core API call — Anthropic format
    # ─────────────────────────────────────
    def _call_api(self, prompt: str) -> str:
        """Make HTTP request to Anthropic-native /v1/messages endpoint."""
        url = f"{self.base_url}/v1/messages"

        headers = {
            "x-api-key": self.key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "max_tokens": 500,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=120)

        # ═══════════════════════════════════
        # DEBUG: Print response details
        # ═══════════════════════════════════
        if self.debug:
            print(f"\n    [DEBUG] Status Code: {resp.status_code}")
            print(f"    [DEBUG] Content-Type: {resp.headers.get('Content-Type', 'N/A')}")
            print(f"    [DEBUG] Raw Text (first 500 chars):")
            print(f"    {resp.text[:500]}")
            print()

        resp.raise_for_status()

        # ── Parse response ──
        try:
            data = resp.json()
        except json.JSONDecodeError:
            if self.debug:
                print(f"    [DEBUG] Response is NOT JSON, returning raw text")
            return resp.text

        if self.debug:
            print(f"    [DEBUG] Response type: {type(data)}")
            if isinstance(data, dict):
                print(f"    [DEBUG] Top-level keys: {list(data.keys())}")

        # ── Format 1: Anthropic native ──
        # {
        #   "id": "msg_...",
        #   "type": "message",
        #   "role": "assistant",
        #   "content": [{"type": "text", "text": "..."}],
        #   "model": "claude-...",
        #   "stop_reason": "end_turn",
        #   "usage": {"input_tokens": ..., "output_tokens": ...}
        # }
        if isinstance(data, dict) and "content" in data:
            content = data["content"]
            if isinstance(content, list) and len(content) > 0:
                if isinstance(content[0], dict) and "text" in content[0]:
                    return content[0]["text"]
                return str(content[0])
            return str(content)

        # ── Format 2: OpenAI compatible (fallback) ──
        if isinstance(data, dict) and "choices" in data:
            msg = data["choices"][0]
            if isinstance(msg, dict) and "message" in msg:
                return msg["message"]["content"]
            if isinstance(msg, dict) and "text" in msg:
                return msg["text"]
            return str(msg)

        # ── Format 3: Direct message ──
        if isinstance(data, dict) and "message" in data:
            msg = data["message"]
            if isinstance(msg, dict) and "content" in msg:
                return msg["content"]
            return str(msg)

        # ── Format 4: Response / result / output fields ──
        if isinstance(data, dict) and "response" in data:
            return str(data["response"])
        if isinstance(data, dict) and "result" in data:
            return str(data["result"])
        if isinstance(data, dict) and "output" in data:
            return str(data["output"])

        # ── Format 5: Raw string ──
        if isinstance(data, str):
            return data

        # ── Fallback ──
        print(f"    ⚠️ Unknown response format. Full response:")
        print(f"    {json.dumps(data, ensure_ascii=False)[:500]}")
        return json.dumps(data)

    # ─────────────────────────────────────
    #  Parse JSON from model output
    # ─────────────────────────────────────
    def _parse_json_response(self, raw: str) -> dict:
        """Parse JSON from model response, handling markdown wrappers."""
        if not raw or raw.strip() == "":
            raise json.JSONDecodeError("Empty response", "", 0)

        raw = raw.strip()

        # Remove markdown wrappers
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

        # Find JSON object in response
        if not raw.startswith("{"):
            start = raw.find("{")
            if start != -1:
                raw = raw[start:]

        if not raw.endswith("}"):
            end = raw.rfind("}")
            if end != -1:
                raw = raw[: end + 1]

        return json.loads(raw)

    # ─────────────────────────────────────
    #  Analyze a single article
    # ─────────────────────────────────────
    def analyze_single(self, article: dict) -> dict:
        """Analyze one article."""
        text = clean_article_text(article)
        prompt = build_sentiment_prompt(text)

        raw_response = None  # Define before try so it's always available

        try:
            raw_response = self._call_api(prompt)
            result = self._parse_json_response(raw_response)

            article["claude_sentiment"] = result["sentiment"]
            article["claude_confidence"] = result["confidence"]
            article["claude_reasoning_ar"] = result["reasoning_ar"]
            article["claude_reasoning_en"] = result["reasoning_en"]
            article["claude_risk_signals"] = result["risk_signals"]
            article["claude_key_events"] = result["key_financial_events"]
            article["claude_impact_duration"] = result["impact_duration"]
            article["claude_numeric"] = self.LABEL_MAP.get(result["sentiment"], 0)

            # Turn off debug after first success
            self.debug = False

        except json.JSONDecodeError as e:
            print(f"    ⚠️ JSON parse error: {e}")
            if raw_response:
                print(f"    Raw response: '{raw_response[:300]}'")
            article["claude_sentiment"] = "error"
            article["claude_confidence"] = 0
            article["claude_reasoning_ar"] = ""
            article["claude_reasoning_en"] = "JSON error"
            article["claude_risk_signals"] = []
            article["claude_key_events"] = []
            article["claude_impact_duration"] = ""
            article["claude_numeric"] = 0

        except requests.exceptions.HTTPError as e:
            print(f"    ⚠️ HTTP error: {e}")
            print(
                f"    Response: {e.response.text[:300] if e.response else 'N/A'}"
            )
            article["claude_sentiment"] = "error"
            article["claude_confidence"] = 0
            article["claude_reasoning_ar"] = ""
            article["claude_reasoning_en"] = str(e)
            article["claude_risk_signals"] = []
            article["claude_key_events"] = []
            article["claude_impact_duration"] = ""
            article["claude_numeric"] = 0

        except Exception as e:
            print(f"    ⚠️ Error: {type(e).__name__}: {e}")
            article["claude_sentiment"] = "error"
            article["claude_confidence"] = 0
            article["claude_reasoning_ar"] = ""
            article["claude_reasoning_en"] = str(e)
            article["claude_risk_signals"] = []
            article["claude_key_events"] = []
            article["claude_impact_duration"] = ""
            article["claude_numeric"] = 0

        return article

    # ─────────────────────────────────────
    #  Analyze all articles for a company
    # ─────────────────────────────────────
    def analyze_company_file(self, input_path: str, output_path: str) -> dict:
        """Analyze all articles for one company."""
        data = load_articles(input_path)
        articles = data["articles"]
        print(f"  📰 Analyzing {len(articles)} articles with Claude...")

        enriched = []
        for i, art in enumerate(tqdm(articles, desc="  Claude")):
            enriched.append(self.analyze_single(art))
            time.sleep(CLAUDE_DELAY)

        data["articles"] = enriched
        dist = {
            "positive": sum(1 for a in enriched if a.get("claude_numeric") == 1),
            "neutral": sum(1 for a in enriched if a.get("claude_numeric") == 0),
            "negative": sum(
                1 for a in enriched if a.get("claude_numeric") == -1
            ),
        }
        data["metadata"]["claude_sentiment"] = {
            "model": self.model,
            "endpoint": self.base_url,
            "analyzed_at": datetime.now().isoformat(),
            "total_analyzed": len(enriched),
            "distribution": dist,
        }

        save_results(data, output_path)
        return data


# ═══════════════════════════════════════════
#  Quick test
# ═══════════════════════════════════════════
if __name__ == "__main__":
    print("🧪 Testing Claude analyzer with DEBUG mode...\n")
    analyzer = ClaudeAnalyzer()

    print("\n🧪 Sending test request...\n")
    test = {
        "title": "أبو قير للأسمدة تقرر توزيع كوبون نقدي على المساهمين",
        "body": "أعلنت شركة أبو قير للأسمدة عن توزيع كوبون نقدي قدره 2.30 جنيه لكل سهم",
    }

    result = analyzer.analyze_single(test.copy())

    print(f"\n{'═' * 50}")
    print(f"📰 Title:      {result['title']}")
    print(f"🎯 Sentiment:  {result.get('claude_sentiment', 'N/A')}")
    print(f"📊 Confidence: {result.get('claude_confidence', 'N/A')}")
    print(f"💬 EN: {result.get('claude_reasoning_en', 'N/A')}")
    print(f"💬 AR: {result.get('claude_reasoning_ar', 'N/A')}")
    print(f"⚡ Risk:       {result.get('claude_risk_signals', 'N/A')}")
    print(f"📅 Duration:   {result.get('claude_impact_duration', 'N/A')}")
    print(f"🔢 Numeric:    {result.get('claude_numeric', 'N/A')}")