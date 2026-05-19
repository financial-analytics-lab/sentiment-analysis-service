# test_claude.py
"""
Minimal probe of the Claude endpoint configured in config.py.

Sends a one-line "say hi" message and reports:
  - HTTP status
  - response body (or first 800 chars)
  - whether the assistant returned a text block
  - approximate latency

Run:  python test_claude.py
"""

import json
import time
import sys

import requests

from config import CLAUDE_API_KEY, CLAUDE_BASE_URL, CLAUDE_MODEL


def _print_response_debug(resp: requests.Response) -> None:
    print("-" * 60)
    print(f"HTTP {resp.status_code}   reason={resp.reason!r}")
    print(f"Final URL: {resp.url}")
    print(f"Elapsed  : {resp.elapsed.total_seconds():.2f}s")

    interesting_headers = [
        "content-type",
        "content-length",
        "server",
        "date",
        "retry-after",
        "x-request-id",
        "request-id",
        "cf-ray",
        "x-trace-id",
        "x-amzn-requestid",
        "x-amz-request-id",
    ]
    print("Response headers:")
    found_any = False
    for name in interesting_headers:
        value = resp.headers.get(name)
        if value is not None:
            found_any = True
            print(f"  {name}: {value}")
    if not found_any:
        print("  <no common diagnostic headers found>")

    print("Body preview:")
    body_text = resp.text or ""
    try:
        body = resp.json()
    except ValueError:
        print(body_text[:2000] if body_text else "<empty body>")
        return

    print(json.dumps(body, ensure_ascii=False, indent=2)[:4000])
    if isinstance(body, dict):
        error_fields = {
            k: body[k]
            for k in body.keys()
            if k.lower() in {"error", "message", "detail", "details", "type", "title", "code"}
        }
        if error_fields:
            print("Error fields:")
            print(json.dumps(error_fields, ensure_ascii=False, indent=2))


def main():
    url = f"{CLAUDE_BASE_URL.rstrip('/')}/v1/messages"
    payload = {
        "model":      CLAUDE_MODEL,
        "max_tokens": 1000,
        "messages":   [{"role": "user", "content": "Say 'hello' in one word."}],
    }
    headers = {
        "x-api-key":         CLAUDE_API_KEY,
        "anthropic-version": "2023-06-01",
        "Content-Type":      "application/json",
    }

    print(f"Endpoint : {url}")
    print(f"Model    : {CLAUDE_MODEL}")
    print(f"Key      : {CLAUDE_API_KEY[:12]}...{CLAUDE_API_KEY[-4:]}")
    print(f"Prompt   : {payload['messages'][0]['content']!r}")
    print("-" * 60)

    t0 = time.perf_counter()
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
    except requests.exceptions.Timeout:
        print(f"TIMEOUT after {time.perf_counter() - t0:.1f}s -- proxy never responded.")
        sys.exit(2)
    except requests.exceptions.ConnectionError as exc:
        print(f"CONNECTION ERROR after {time.perf_counter() - t0:.1f}s: {exc}")
        sys.exit(2)
    dt = time.perf_counter() - t0

    print(f"HTTP {resp.status_code}   latency={dt:.2f}s")
    try:
        resp.raise_for_status()
    except requests.exceptions.HTTPError:
        _print_response_debug(resp)
        print(f"FAIL: endpoint returned HTTP {resp.status_code}.")
        sys.exit(1)

    print("-" * 60)

    body_text = resp.text
    try:
        body = resp.json()
        print(json.dumps(body, ensure_ascii=False, indent=2)[:1200])
    except ValueError:
        print("Body is not JSON. First 800 chars:")
        print(body_text[:800])
        sys.exit(1)

    print("-" * 60)

    content = body.get("content")
    if isinstance(content, list):
        texts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
        if texts:
            print(f"OK: assistant text = {texts[0]!r}")
            sys.exit(0)
        print("FAIL: 200 OK but no text block in content.")
        sys.exit(1)

    if isinstance(body.get("choices"), list):  # OpenAI-compatible proxy
        msg = body["choices"][0].get("message", {})
        print(f"OK (OpenAI-shape): assistant text = {msg.get('content')!r}")
        sys.exit(0)

    print("FAIL: 200 OK but response shape is unrecognised.")
    sys.exit(1)


if __name__ == "__main__":
    main()
