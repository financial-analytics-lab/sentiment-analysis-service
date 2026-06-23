"""
build_context: dual ticker + sector quantitative context for the agent.

Wraps features.compute_context (ticker-level TA) and adds a sector_context
block with sector-level TA signals. The sector block uses ALL sector members
(including the target ticker) because we are predicting the sector index,
not computing an abnormal-return benchmark.

The ticker-excl-target calculation is only needed in evaluate.py when
deriving the implied_abnormal ground truth — not in the prediction path.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from features import (  # noqa: E402
    _atr_pct,
    _build_close_df,
    _load_all_prices,
    _pct_return,
    _rsi,
    compute_context as _core_compute_context,
    news_in_trading_hours,
)

from .config import (
    ACTIVE_HORIZONS,
    sector_neutral_band,
    ticker_magnitude_bands,
    ticker_neutral_band,
)


def build_context(ticker: str, event_date: str, article: dict | None = None) -> dict:
    """
    Return a dual-context dict for the agent.

    Fields:
      ticker_context   — price state, TA signals, per-company bands
      sector_context   — sector-level TA signals, RSI, momentum, bands
      news_timing      — in_trading_hours flag (controls eval window start)
    """
    base = _core_compute_context(ticker, event_date)

    raw = _load_all_prices()
    sector_name = raw[ticker]["sector"]
    close_df = _build_close_df(raw, event_date)

    # --- Sector series (ALL members, including target) ---------------------
    sector_tickers = [
        t for t, d in raw.items()
        if d["sector"] == sector_name and t in close_df.columns
    ]
    sector_df = close_df[sector_tickers].dropna(how="all")
    sector_series = sector_df.mean(axis=1).dropna()

    sector_rsi_14  = _rsi(sector_series, 14)
    sector_atr_14d = _atr_pct(sector_series, 14)

    sector_ma_50  = (
        round(float(sector_series.iloc[-50:].mean()), 4)
        if len(sector_series) >= 50 else None
    )
    sector_ma_200 = (
        round(float(sector_series.iloc[-200:].mean()), 4)
        if len(sector_series) >= 200 else None
    )
    sector_last = float(sector_series.iloc[-1]) if not sector_series.empty else None

    sector_mom_5d  = (
        round((_pct_return(sector_series, 5)  or 0.0), 2)
        if len(sector_series) > 5 else None
    )
    sector_mom_20d = (
        round((_pct_return(sector_series, 20) or 0.0), 2)
        if len(sector_series) > 20 else None
    )

    sector_rsi_zone = "unknown"
    if sector_rsi_14 is not None:
        if sector_rsi_14 > 70:
            sector_rsi_zone = "overbought"
        elif sector_rsi_14 < 30:
            sector_rsi_zone = "oversold"
        else:
            sector_rsi_zone = "neutral"

    # --- Per-horizon bands (using agent horizons, not features horizons) --
    ticker_atr = base["technical_signals"]["atr_pct_14d"]

    ticker_bands: dict = {}
    sector_bands: dict = {}
    for h in ACTIVE_HORIZONS:
        ticker_bands[h] = {
            "neutral_band_pct": ticker_neutral_band(ticker_atr, h),
            "magnitude_bands_pct": ticker_magnitude_bands(ticker_atr, h),
        }
        sector_bands[h] = {
            "neutral_band_pct": sector_neutral_band(sector_atr_14d, h),
        }

    # --- News timing -------------------------------------------------------
    timing: dict = {}
    if article is not None:
        in_hours = news_in_trading_hours(article)
        timing = {
            "in_trading_hours": in_hours,
            "arrived_at_str": (
                article.get("datetime")
                or f"{article.get('date', event_date)} {article.get('time', '?')}"
            ),
            "start_day_label": (
                "event_date (in trading hours)"
                if in_hours
                else "next trading day (outside hours)"
            ),
        }

    return {
        "ticker":               ticker,
        "company":              base["company"],
        "sector":               sector_name,
        "business_description": base["business_description"],
        "event_date":           event_date,
        "last_price_date":      base["last_price_date"],
        "news_timing":          timing,
        "ticker_context": {
            "price_state":             base["price_state"],
            "technical_signals":       base["technical_signals"],
            "market_state":            base["market_state"],
            "recent_abnormal_returns": base["recent_abnormal_returns"],
            "atr_14d":                 ticker_atr,
            "horizon_bands":           ticker_bands,
        },
        "sector_context": {
            "sector_name":      sector_name,
            "sector_members":   sector_tickers,
            "n_members":        len(sector_tickers),
            "level":            round(sector_last, 2) if sector_last else None,
            "ma_50":            sector_ma_50,
            "ma_200":           sector_ma_200,
            "rsi_14":           sector_rsi_14,
            "rsi_zone":         sector_rsi_zone,
            "momentum_5d_pct":  sector_mom_5d,
            "momentum_20d_pct": sector_mom_20d,
            "atr_14d":          sector_atr_14d,
            "horizon_bands":    sector_bands,
        },
    }
