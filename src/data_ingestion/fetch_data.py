import os
import yfinance as yf
import pandas as pd
from datetime import datetime

def fetch_and_save_tickers(ticker: str, start: str, end: str, saved_dir)-> pd.DataFrame:
    """
    Fetch daily close prices for a given ticker from Yahoo Finance
    for a given [start; end] period
    and save as CSV in the data/ directory.
    """
    print(f"Fetching {ticker} from {start} to {end}...")
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)

    if df.empty:
        print(f"No data returned for {ticker}")
        return pd.DataFrame()

    # if multiindex, flatten it
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Select the 'Close' column
    df = df[["Close"]].rename(columns={"Close": ticker})
    df.index.name = "date"

    # Save individual CSV
    filename = os.path.join(saved_dir, f"{ticker}_sample.csv")
    df.to_csv(filename, index=True)
    print(f"Saved {ticker} data to {filename} ({df.shape[0]} rows)")
    return df
