import pandas as pd

returns = pd.read_parquet('returns_daily.parquet')
returns.index = pd.to_datetime(returns.index)
sector_returns = pd.read_parquet('sector_returns_daily.parquet')
sector_returns.index = pd.to_datetime(sector_returns.index)

cases = [
    ('JUFO', 'Food & Beverages', '2025-06-25', 20),
    ('GBCO', 'Diversified Holding', '2025-06-23', 20),
]

for ticker, sector, event_date, n in cases:
    print('\n' + '='*60)
    print(f'Ticker: {ticker}  Sector: {sector}  Event date: {event_date}')
    d = pd.Timestamp(event_date)
    incl = True
    if d not in returns.index:
        print('Event date not in returns index')
    # show the next 5 trading days from event_date inclusive
    future = returns.loc[returns.index >= d].head(5)
    print('\nStock returns:')
    if ticker in future.columns:
        print(future[[ticker]])
    else:
        print('Ticker not in returns.columns')
    print('\nSector returns:')
    if sector in sector_returns.columns:
        sec_future = sector_returns.loc[future.index]
        print(sec_future[[sector]])
    else:
        print('Sector not in sector_returns.columns')
    # compute abnormal if possible
    if ticker in future.columns and sector in sector_returns.columns:
        sec_window = sector_returns.loc[future.index, sector]
        stock_window = future[ticker]
        ar = stock_window - sec_window
        print('\nAbnormal returns:')
        print(ar)
        print('\nSum AR (cum):', ar.sum())
    print('='*60)
