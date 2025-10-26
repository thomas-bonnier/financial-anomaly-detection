import os
from src.data_ingestion.fetch_data import fetch_and_save_tickers
from src.preprocessing.preprocess_data import preprocess_multivariate
from src.preprocessing.dataset import TimeSeriesDataset
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Directory to save CSVs
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Example: Tech sector tickers
SECTOR = "tech"
TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA"]

# Training period
START_DATE = "2020-01-01"
END_DATE = "2025-12-31"

def main():
    # Fetch data
    all_dfs = []
    for ticker in TICKERS:
        df = fetch_and_save_tickers(ticker, START_DATE, END_DATE, DATA_DIR)
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        print("No data fetched. Exiting.")
        return

    # Merge all tickers on date index
    combined_df = pd.concat(all_dfs, axis=1, join="inner").sort_index()

    # Save combined dataset
    combined_file = os.path.join(DATA_DIR, f"combined_stocks_{SECTOR}.csv")
    combined_df.to_csv(combined_file, index=True)
    print(f"Saved combined dataset to {combined_file} with shape {combined_df.shape[0]}")
    
    # Preprocess dataset
    filepath = os.path.join(DATA_DIR, "combined_stocks_tech.csv")
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    X, scaler = preprocess_multivariate(df, window_size=30)
    
    # Create torch dataset and data loader
    dataset = TimeSeriesDataset(X)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    print(next(iter(loader)))
    print(next(iter(loader)).shape)
    

if __name__ == "__main__":
    main()