# EGX News-Impact Agent v2 — API Reference

Base URL: `http://localhost:8001`

All requests and responses use `application/json`.

---

## Endpoints

### `GET /health`

Liveness check. Returns immediately with no LLM calls.

**Response `200 OK`**

```json
{
  "status": "ok",
  "version": "2.0.0"
}
```

---

### `POST /analyze`

Analyze an Arabic news article about an EGX-listed company and return dual-target
directional predictions (ticker absolute return direction + sector return direction)
for each active horizon.

**LLM calls triggered:** 3 on the hot path (Sentiment + Expert + Analyst), 4 if
the Critic gate fires (≈ 20–30% of requests).

---

#### Request body

```json
{
  "ticker": "COMI",
  "title": "البنك التجاري الدولي يعلن توزيع أرباح نقدية بواقع 3 جنيهات للسهم",
  "body": "أعلن البنك التجاري الدولي-مصر (كومي) اليوم عن توزيع أرباح نقدية...",
  "published_at": "2025-03-15"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `ticker` | string | yes | EGX ticker symbol (e.g. `"COMI"`, `"ETEL"`) |
| `title` | string | yes | Arabic headline |
| `body` | string | yes | Arabic article body |
| `published_at` | string | yes | Publication date `YYYY-MM-DD` or full ISO datetime |
| `source` | string | no | News source label (informational, not used in prediction) |

**Alternative envelope shape** — wrap in `"article"` key, optionally override ticker at top level:

```json
{
  "ticker": "COMI",
  "article": {
    "ticker": "COMI",
    "title": "...",
    "body": "...",
    "published_at": "2025-03-15"
  }
}
```

---

#### Response `200 OK`

```json
{
  "status": "ok",
  "ticker": "COMI",
  "event_type": "dividend",
  "processed_at": "2025-03-15T09:15:42.123456+00:00",
  "critic_invoked": false,
  "outlook": {
    "short": {
      "horizon_days": 5,
      "ticker": {
        "direction": "up",
        "magnitude": "small",
        "confidence": 0.72,
        "reasoning": "أعلن البنك عن توزيع أرباح يعكس قوة الأرباح..."
      },
      "sector": {
        "direction": "neutral",
        "magnitude": "none",
        "confidence": 0.63,
        "reasoning": "الحدث خاص بالشركة ولا يشير إلى تحرك قطاعي..."
      },
      "implied_abnormal": "up"
    },
    "medium": {
      "horizon_days": 10,
      "ticker": {
        "direction": "up",
        "magnitude": "small",
        "confidence": 0.65,
        "reasoning": "..."
      },
      "sector": {
        "direction": "up",
        "magnitude": "small",
        "confidence": 0.58,
        "reasoning": "..."
      },
      "implied_abnormal": "neutral"
    }
  },
  "summary": "أعلن البنك التجاري الدولي-مصر عن توزيع أرباح نقدية...",
  "technical_view": "يشير مؤشر القوة النسبية (RSI) إلى مستوى 68...",
  "risks": "قد يؤدي ارتفاع أسعار الفائدة إلى تغيير توقعاتنا..."
}
```

---

#### Response field reference

**Top-level**

| Field | Type | Description |
|---|---|---|
| `status` | `"ok"` | Always `"ok"` on success |
| `ticker` | string | EGX ticker symbol |
| `event_type` | string | Classified event type (see Event Types below) |
| `processed_at` | ISO datetime | UTC timestamp of processing |
| `critic_invoked` | boolean | `true` if the Critic LLM was called (gate triggered) |
| `outlook` | object | Per-horizon predictions (keys: `"short"`, `"medium"`) |
| `summary` | string | Arabic plain-language news summary for retail investors |
| `technical_view` | string | Arabic TA commentary with inline Arabic term explanations |
| `risks` | string | Arabic description of key uncertainties that could reverse the prediction |

**`outlook[horizon]`**

| Field | Type | Description |
|---|---|---|
| `horizon_days` | integer | Number of trading days for this horizon (`5` or `10`) |
| `ticker` | LegPrediction | Ticker absolute return direction prediction |
| `sector` | LegPrediction | Sector index return direction prediction |
| `implied_abnormal` | `"up"` \| `"down"` \| `"neutral"` | Derived abnormal return direction (ticker vs sector) |

**`LegPrediction`**

| Field | Type | Description |
|---|---|---|
| `direction` | `"up"` \| `"down"` \| `"neutral"` | Predicted return direction |
| `magnitude` | `"small"` \| `"medium"` \| `"large"` \| `"none"` | Expected move size relative to ATR bands. `"none"` only when direction is `"neutral"` |
| `confidence` | float 0–1 | Raw LLM confidence. Treat as a qualitative indicator, not a calibrated probability |
| `reasoning` | string | Arabic explanation for this leg's prediction |

**`implied_abnormal` derivation rules** (deterministic, no additional LLM call):

| Ticker | Sector | `implied_abnormal` |
|---|---|---|
| `up` | `neutral` or `down` | `up` (outperformance) |
| `down` | `neutral` or `up` | `down` (underperformance) |
| `up` | `up`, ticker magnitude > sector magnitude | `up` |
| `down` | `down`, ticker magnitude > sector magnitude | `down` |
| Same direction, same magnitude | — | `neutral` (wash) |
| `neutral` (either leg) | — | `neutral` |

---

#### Event types

| Value | Meaning |
|---|---|
| `dividend` | Cash or stock dividend announcement |
| `capital_increase` | Rights issue, capital raise |
| `earnings` | Earnings report (profit, loss, revenue) |
| `regulatory` | Regulatory approval, fine, compliance news |
| `operational` | Contract win/loss, capacity change, operational update |
| `m_and_a` | Merger, acquisition, takeover bid |
| `macro` | Macroeconomic event affecting the market broadly |
| `other` | Does not fit the above categories |

---

#### Horizon definitions

| Horizon | Trading days | Use case |
|---|---|---|
| `short` | 5 (≈ 1 week) | Immediate post-announcement reaction |
| `medium` | 10 (≈ 2 weeks) | Short-term drift after the announcement effect settles |

The 20-day "large" horizon is excluded from v2 — empirically a coin flip (0.467 directional accuracy).

---

#### Magnitude bands

Magnitude bands are calibrated per ticker using a 14-day average absolute return proxy (ATR):

| Band | Return range |
|---|---|
| `neutral` | Within ±0.5 × ATR × √N |
| `small` | 0.5×–0.7× ATR × √N |
| `medium` | 0.7×–2× ATR × √N |
| `large` | > 2× ATR × √N |

where N = horizon trading days.

---

#### Error responses

| Status | When |
|---|---|
| `400 Bad Request` | Missing required fields, invalid JSON, or unknown ticker |
| `500 Internal Server Error` | LLM call failed after retries, or unexpected pipeline error |

**Error body:**

```json
{
  "detail": "Missing required fields: ['body', 'ticker']"
}
```

---

## Supported tickers

The agent covers all ~31 EGX-listed tickers for which price history is available:

`ABUK` `VLMRA` `VLMR` `SKPC` `AMOC` `MFPC` `COMI` `ADIB` `CIEB` `BTFH`
`HRHO` `CCAP` `FWRY` `EFIH` `RAYA` `PHDC` `MASR` `TMGH` `EMFD` `ORHD`
`ARCC` `MCQE` `ORAS` `RMDA` `ISPH` `JUFO` `EGAL` `ETEL` `ORWE` `EAST` `GBCO`

Passing an unknown ticker returns `400 Bad Request`.

---

## Interactive docs

While the service is running:

- **Swagger UI** (try requests in-browser): `http://localhost:8001/docs`
- **ReDoc** (clean reference): `http://localhost:8001/redoc`

---

## Example: full cURL request + response

```bash
curl -s -X POST http://localhost:8001/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "ETEL",
    "title": "المصرية للاتصالات تفوز بعقد بنية تحتية بقيمة 500 مليون جنيه",
    "body": "أعلنت شركة المصرية للاتصالات اليوم عن فوزها بعقد لإنشاء بنية تحتية للاتصالات في المناطق الجديدة بقيمة 500 مليون جنيه مصري. يمتد العقد لمدة 3 سنوات.",
    "published_at": "2025-04-10"
  }'
```

**Response (abbreviated):**

```json
{
  "status": "ok",
  "ticker": "ETEL",
  "event_type": "operational",
  "processed_at": "2025-04-10T10:30:00.000000+00:00",
  "critic_invoked": false,
  "outlook": {
    "short": {
      "horizon_days": 5,
      "ticker": {
        "direction": "up",
        "magnitude": "small",
        "confidence": 0.68,
        "reasoning": "الفوز بعقد ضخم يعزز الإيرادات المتوقعة..."
      },
      "sector": {
        "direction": "neutral",
        "magnitude": "none",
        "confidence": 0.65,
        "reasoning": "الحدث محدود تأثيره على قطاع الاتصالات ككل..."
      },
      "implied_abnormal": "up"
    },
    "medium": {
      "horizon_days": 10,
      "ticker": {
        "direction": "up",
        "magnitude": "small",
        "confidence": 0.61,
        "reasoning": "..."
      },
      "sector": {
        "direction": "neutral",
        "magnitude": "none",
        "confidence": 0.60,
        "reasoning": "..."
      },
      "implied_abnormal": "up"
    }
  },
  "summary": "فازت المصرية للاتصالات بعقد بنية تحتية بقيمة 500 مليون جنيه...",
  "technical_view": "يتداول السهم بالقرب من المتوسط المتحرك لـ 50 يومًا...",
  "risks": "أي تأخير في تنفيذ العقد أو تراجع الطلب قد يؤثر على التقديرات..."
}
```
