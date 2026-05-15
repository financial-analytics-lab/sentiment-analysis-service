"""
Convert egyptian_stocks_2020_2025.json into the two parquet files
required by evaluate.py:

  returns_daily.parquet        — log returns, index=date, columns=tickers
  sector_returns_daily.parquet — log returns, index=date, columns=sector names
                                 (equal-weighted average across sector members)

Usage:
  python prepare_data.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

SECTOR_MAP = {
    "ABUK": "Petroleum & Petrochemicals", "SKPC": "Petroleum & Petrochemicals",
    "AMOC": "Petroleum & Petrochemicals", "MFPC": "Petroleum & Petrochemicals",
    "EKHO": "Petroleum & Petrochemicals",
    "COMI": "Banking", "ADIB": "Banking", "CIEB": "Banking",
    "HRHO": "Non-Bank Financials", "CCAP": "Non-Bank Financials",
    "BTFH": "Non-Bank Financials",
    "FWRY": "Technology & Fintech", "EFIH": "Technology & Fintech",
    "RAYA": "Technology & Fintech",
    "PHDC": "Real Estate", "MASR": "Real Estate", "TMGH": "Real Estate",
    "EMFD": "Real Estate", "ORHD": "Real Estate",
    "ARCC": "Construction & Building Materials",
    "MCQE": "Construction & Building Materials",
    "ORAS": "Construction & Building Materials",
    "RMDA": "Healthcare & Pharmaceuticals", "ISPH": "Healthcare & Pharmaceuticals",
    "JUFO": "Food & Beverages",
    "EGAL": "Metals & Mining",
    "ETEL": "Telecommunications",
    "ORWE": "Textiles",
    "EAST": "Tobacco & Consumer Goods",
    "GBCO": "Diversified Holding",
}

JSON_PATH = Path("egyptian_stocks_2020_2025.json")
OUT_DIR   = Path(".")

def main():
    print("Loading JSON...")
    with open(JSON_PATH, encoding="utf-8") as f:
        data = json.load(f)

    # Build a close-price DataFrame: index=date, columns=tickers
    closes = {}
    for ticker, info in data.items():
        prices = info.get("prices", [])
        if not prices:
            print(f"  [WARN] {ticker}: no prices, skipping")
            continue
        s = pd.Series(
            {p["Date"]: p["Close"] for p in prices},
            name=ticker,
            dtype=float,
        )
        closes[ticker] = s

    close_df = pd.DataFrame(closes)
    close_df.index = pd.to_datetime(close_df.index)
    close_df.sort_index(inplace=True)
    print(f"Close prices shape: {close_df.shape}  ({close_df.index[0].date()} to {close_df.index[-1].date()})")

    # Log returns: ln(P_t / P_{t-1})
    returns = np.log(close_df / close_df.shift(1))

    # Sector returns: equal-weighted mean of members present on each day
    sectors = {}
    for sector in set(SECTOR_MAP.values()):
        members = [t for t, s in SECTOR_MAP.items() if s == sector and t in returns.columns]
        if not members:
            print(f"  [WARN] sector '{sector}': no members found in data")
            continue
        sectors[sector] = returns[members].mean(axis=1)

    sector_returns = pd.DataFrame(sectors)
    sector_returns.index = returns.index

    # Save
    returns_path = OUT_DIR / "returns_daily.parquet"
    sector_path  = OUT_DIR / "sector_returns_daily.parquet"
    returns.to_parquet(returns_path)
    sector_returns.to_parquet(sector_path)

    print(f"Wrote: {returns_path}  ({returns.shape})")
    print(f"Wrote: {sector_path}  ({sector_returns.shape})")
    print("Done. Now run:")
    print(
        "  python evaluate.py "
        "--predictions-dir outputs/predictions "
        "--returns returns_daily.parquet "
        "--sector-returns sector_returns_daily.parquet "
        "--out outputs/evaluation"
    )

if __name__ == "__main__":
    main()
