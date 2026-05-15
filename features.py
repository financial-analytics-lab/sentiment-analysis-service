# features.py
"""
Computes feature context for an EGX 30 ticker as of a given event date.
No lookahead: only price data with Date <= event_date is used.
"""

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PRICES_FILE = Path(__file__).parent / "egyptian_stocks_2020_2025.json"

# One-line descriptions grounding the LLM in what each company actually does.
COMPANY_DESCRIPTIONS: dict[str, str] = {
    "ABUK":  "Abou Kir Fertilizers — Egypt's largest nitrogen fertilizer (urea/ammonium nitrate) producer near Alexandria; exports to Europe; earnings sensitive to natural gas feedstock price and EUR/USD.",
    "VLMRA": "Egyptian Kuwaiti Holding (EKHOA preferred class, rebranded Valmore Holding post-2026) — diversified conglomerate with exposure to fertilizers, cement, and industrial investments; EGP-denominated preferred share class.",
    "VLMR":  "Egyptian Kuwaiti Holding (EKHO ordinary, rebranded Valmore Holding post-2026) — diversified conglomerate with exposure to fertilizers, cement, construction, and industrial holdings; Kuwaiti-Egyptian joint venture.",
    "PHDC":  "Palm Hills Developments — upscale mixed-use real estate developer with compounds in New Cairo, 6th of October, and the North Coast; revenues in EGP tied to pre-sales velocity.",
    "EFIH":  "E-Finance for Digital and Financial Investments — Egypt's national digital payments backbone processing government collections; partially state-owned; high-margin, near-monopoly revenues.",
    "ARCC":  "Arabian Cement Company — major cement producer in the Suez corridor; exposed to construction activity, energy costs, and domestic cement prices.",
    "MASR":  "Madinet Masr for Housing and Development — listed real estate developer focused on integrated urban communities in Greater Cairo (Sarai, Taj City).",
    "MFPC":  "Misr Fertilizers Production Company (MOPCO) — urea and melamine producer in Damietta using GASCO natural gas; ~60 % revenues from USD-denominated urea exports.",
    "COMI":  "Commercial International Bank (CIB) — Egypt's largest private-sector bank by assets and net income; widely used as a bellwether for Egyptian banking sector health.",
    "RMDA":  "Rameda — Egyptian generic pharmaceutical manufacturer; domestic-focused revenues in EGP; sensitive to regulatory pricing decisions.",
    "ADIB":  "Abu Dhabi Islamic Bank Egypt — Islamic retail and corporate banking subsidiary of ADIB UAE; serves SMEs and retail depositors under Sharia-compliant products.",
    "ORWE":  "Oriental Weavers — world's largest machine-made carpet manufacturer; ~70 % revenues from exports priced in USD and EUR; major natural gas consumer.",
    "CCAP":  "QALA for Financial Investments (formerly Cairo Capital) — non-bank financial services and investment holding company.",
    "HRHO":  "EFG Holding (formerly EFG Hermes) — Egypt and MENA's leading investment bank: brokerage, asset management, investment banking, and consumer finance (Valu BNPL).",
    "CIEB":  "Credit Agricole Egypt — French-owned mid-tier retail and corporate bank; strong SME and trade finance franchise in Egypt.",
    "GBCO":  "GB Corp — diversified holding group with exposure to financial services, consumer goods, and industrial businesses.",
    "JUFO":  "Juhayna Food Industries — Egypt's leading dairy, juice, and cooking oil producer; wide nationwide retail distribution; defensive consumer staples.",
    "TMGH":  "TMG Holding (Talaat Moustafa Group) — Egypt's largest listed real estate developer by land bank; flagship project is Madinaty new city east of Cairo.",
    "ISPH":  "Ibnsina Pharma — Egypt's largest private pharmaceutical wholesale distributor, serving pharmacies and hospitals; thin-margin, high-volume model.",
    "EMFD":  "Emaar Misr for Development — premium real estate developer; wholly-owned subsidiary of Emaar Properties (Dubai); projects include Uptown Cairo and Marassi.",
    "ORAS":  "Orascom Construction PLC — major EPC contractor with operations in Egypt and the USA (Ceres Environmental); revenues largely USD-denominated from infrastructure and industrial projects.",
    "ORHD":  "Orascom Development Egypt — integrated destination developer combining hotels, residences, and amenities in tourist areas (El Gouna, Makadi, Fayoum); exposure to tourism cycles.",
    "MCQE":  "Misr Cement Qena — cement producer in Upper Egypt; exposed to construction sector cycles and energy cost inflation.",
    "FWRY":  "Fawry for Banking Technology and Electronic Payments — Egypt's largest e-payments and bill-payment platform; processes 4 M+ daily transactions across 300 K+ merchant touchpoints.",
    "EGAL":  "Egypt Aluminum (EGAL) — state-controlled aluminum smelter in Nag Hammadi; energy-intensive; benefits from subsidized electricity; earnings track global LME aluminum prices.",
    "EAST":  "Eastern Company — Egypt's state-affiliated monopoly cigarette manufacturer; defensive stock with stable, recurring revenues and historically high dividend yields.",
    "ETEL":  "Telecom Egypt (WE) — Egypt's fixed-line telecom monopoly, wholesale fiber and submarine-cable provider; high dividend payer with regulated, recurring revenues.",
    "SKPC":  "Sidi Kerir Petrochemicals — Egypt's main polyethylene producer using ethylene from SIDPEC; USD-linked export revenues; earnings sensitive to crude-oil-derived feedstock margins.",
    "RAYA":  "Raya Holding for Financial Investments — diversified group: Raya Contact Center (BPO), consumer electronics distribution, and manufacturing subsidiaries across Africa and the Middle East.",
    "BTFH":  "Beltone Holdings — Egyptian investment banking boutique offering equity brokerage, sell-side research, and asset management services.",
    "AMOC":  "Alexandria Mineral Oils Company (AMOC) — petroleum refinery producing specialty mineral oil fractions and lubricants; earnings track crude oil spreads and downstream pricing.",
}

_price_cache: Optional[dict] = None


def _load_all_prices() -> dict:
    global _price_cache
    if _price_cache is None:
        with open(PRICES_FILE, "r", encoding="utf-8") as f:
            _price_cache = json.load(f)
    return _price_cache


def _build_close_df(raw: dict, up_to_date: str) -> pd.DataFrame:
    """Date-aligned DataFrame of Close prices for every ticker up to event_date."""
    series: dict[str, pd.Series] = {}
    for ticker, info in raw.items():
        rows = [(r["Date"], r["Close"]) for r in info["prices"] if r["Date"] <= up_to_date]
        if rows:
            dates, closes = zip(*rows)
            series[ticker] = pd.Series(
                list(closes), index=pd.to_datetime(list(dates)), name=ticker
            )
    return pd.DataFrame(series).sort_index()


def _pct_return(series: pd.Series, lookback: int) -> Optional[float]:
    if len(series) <= lookback:
        return None
    return round((float(series.iloc[-1]) / float(series.iloc[-1 - lookback]) - 1) * 100, 2)


def _rsi(closes: pd.Series, period: int = 14) -> Optional[float]:
    """RSI using Wilder's exponential smoothing (alpha = 1/period)."""
    if len(closes) < period + 1:
        return None
    delta = closes.diff().dropna()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    return round(float(100 - 100 / (1 + avg_gain / avg_loss)), 2)


def compute_context(ticker: str, event_date: str) -> dict:
    """
    Return a feature-context dict for `ticker` as of `event_date` (YYYY-MM-DD).

    The dict is designed to be serialised directly into LLM prompts.
    All numeric fields are rounded to 2 decimal places.
    No data beyond event_date leaks into any calculation.
    """
    raw = _load_all_prices()

    if ticker not in raw:
        raise ValueError(f"Ticker '{ticker}' not found in price data.")

    close_df = _build_close_df(raw, event_date)

    if ticker not in close_df.columns or close_df[ticker].dropna().empty:
        raise ValueError(f"No price data for '{ticker}' up to {event_date}.")

    t_close = close_df[ticker].dropna()

    # ── Company identity ──────────────────────────────────────────────────────
    info      = raw[ticker]
    sector    = info["sector"]
    company   = info["company"]
    desc      = COMPANY_DESCRIPTIONS.get(ticker, f"{company} — {sector}")

    # ── Price state ───────────────────────────────────────────────────────────
    last_close = round(float(t_close.iloc[-1]), 2)
    last_date  = str(t_close.index[-1].date())

    ret_1d  = _pct_return(t_close, 1)
    ret_5d  = _pct_return(t_close, 5)
    ret_20d = _pct_return(t_close, 20)
    ret_60d = _pct_return(t_close, 60)

    ma50  = round(float(t_close.iloc[-50:].mean()),  2) if len(t_close) >= 50  else None
    ma200 = round(float(t_close.iloc[-200:].mean()), 2) if len(t_close) >= 200 else None

    pct_vs_ma50  = round((last_close / ma50  - 1) * 100, 2) if ma50  else None
    pct_vs_ma200 = round((last_close / ma200 - 1) * 100, 2) if ma200 else None

    log_ret = np.log(t_close / t_close.shift(1)).dropna()

    vol_20d: Optional[float] = None
    if len(log_ret) >= 20:
        vol_20d = round(float(log_ret.iloc[-20:].std() * math.sqrt(252) * 100), 2)

    # Vol regime: compare current 20d vol to the 1-year median of rolling 20d vols.
    vol_regime = "insufficient_data"
    if len(log_ret) >= 252 and vol_20d is not None:
        rolling_vol = log_ret.rolling(20).std() * math.sqrt(252) * 100
        median_1yr  = float(rolling_vol.iloc[-252:].median())
        if vol_20d > median_1yr * 1.2:
            vol_regime = "elevated"
        elif vol_20d < median_1yr * 0.8:
            vol_regime = "low"
        else:
            vol_regime = "normal"

    rsi_14 = _rsi(t_close, 14)

    # ── Market state ──────────────────────────────────────────────────────────
    # Sector: equal-weighted average of all sector peers present in the price file.
    sector_tickers = [
        t for t, d in raw.items()
        if d["sector"] == sector and t in close_df.columns
    ]
    sector_close_df = close_df[sector_tickers].dropna(how="all")
    sector_series   = sector_close_df.mean(axis=1).dropna()
    sector_ret_1d   = _pct_return(sector_series, 1)
    sector_ret_5d   = _pct_return(sector_series, 5)

    # Relative strength: how the stock moved vs its own sector peers.
    rel_strength_vs_sector_1d: Optional[float] = (
        round(ret_1d - sector_ret_1d, 2)
        if ret_1d is not None and sector_ret_1d is not None
        else None
    )

    # ── Correlation state ─────────────────────────────────────────────────────
    ret_df = close_df.pct_change(fill_method=None).dropna(how="all")
    window = ret_df.iloc[-90:] if len(ret_df) >= 90 else ret_df

    t_series = window[ticker].dropna() if ticker in window.columns else pd.Series(dtype=float)

    # Ticker vs sector-index correlation over the 90-day window
    sector_ret_idx = sector_close_df.pct_change(fill_method=None).dropna(how="all").mean(axis=1)
    s_aligned      = sector_ret_idx.reindex(t_series.index)
    common_ts      = t_series.index.intersection(s_aligned.dropna().index)
    sector_corr: Optional[float] = (
        round(float(t_series.loc[common_ts].corr(s_aligned.loc[common_ts])), 2)
        if len(common_ts) > 10
        else None
    )

    # Top 3 peer correlations (excluding self)
    peer_corrs: dict[str, float] = {}
    for peer in window.columns:
        if peer == ticker:
            continue
        p_series = window[peer].dropna()
        common_p = t_series.index.intersection(p_series.index)
        if len(common_p) > 10:
            t_slice = t_series.loc[common_p]
            p_slice = p_series.loc[common_p]
            if t_slice.std() == 0 or p_slice.std() == 0:
                continue
            c = float(t_slice.corr(p_slice))
            if not math.isnan(c):
                peer_corrs[peer] = round(c, 2)

    top_3_peers = [
        {"ticker": t, "correlation": c, "sector": raw[t]["sector"]}
        for t, c in sorted(peer_corrs.items(), key=lambda x: x[1], reverse=True)[:3]
    ]

    # ── Recent abnormal returns (last 5 trading days) ─────────────────────────
    t_daily_ret = (
        ret_df[ticker].dropna() if ticker in ret_df.columns else pd.Series(dtype=float)
    )
    s_daily_ret = sector_close_df.pct_change(fill_method=None).dropna(how="all").mean(axis=1)

    recent_ars: list[dict] = []
    for i in range(min(5, len(t_daily_ret))):
        idx   = t_daily_ret.index[-(i + 1)]
        t_ret = round(float(t_daily_ret.iloc[-(i + 1)]) * 100, 2)
        s_val = s_daily_ret.loc[idx] if idx in s_daily_ret.index else None
        s_ret = round(float(s_val) * 100, 2) if s_val is not None and not math.isnan(float(s_val)) else None
        ar    = round(t_ret - s_ret, 2) if s_ret is not None else None
        recent_ars.append({
            "date":               str(idx.date()),
            "stock_return_pct":   t_ret,
            "sector_return_pct":  s_ret,
            "abnormal_return_pct": ar,
        })
    recent_ars.reverse()  # oldest first

    return {
        "ticker":               ticker,
        "company":              company,
        "sector":               sector,
        "business_description": desc,
        "event_date":           event_date,
        "last_price_date":      last_date,
        "price_state": {
            "last_close":             last_close,
            "return_1d":              ret_1d,
            "return_5d":              ret_5d,
            "return_20d":             ret_20d,
            "return_60d":             ret_60d,
            "pct_vs_ma50":            pct_vs_ma50,
            "pct_vs_ma200":           pct_vs_ma200,
            "vol_20d_annualized_pct": vol_20d,
            "vol_regime":             vol_regime,
            "rsi_14":                 rsi_14,
        },
        "market_state": {
            "sector_return_1d":             sector_ret_1d,
            "sector_return_5d":             sector_ret_5d,
            "relative_strength_vs_sector_1d": rel_strength_vs_sector_1d,
        },
        "correlation_state": {
            "corr_with_sector_90d":   sector_corr,
            "top_3_correlated_peers": top_3_peers,
        },
        "recent_abnormal_returns": recent_ars,
    }


# ── Quick smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pprint
    ctx = compute_context("ABUK", "2025-01-05")
    pprint.pprint(ctx, sort_dicts=False)
