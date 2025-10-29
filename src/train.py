import os
from src.data_ingestion.fetch_data import fetch_and_save_tickers
from src.preprocessing.preprocess_data import preprocess_multivariate
from src.preprocessing.dataset import TimeSeriesDataset
from src.models.autoencoder import Autoencoder
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Directory to save CSVs
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# SECTORS and TICKERS
SECTOR_INDEX = {"tech": 0, "energy": 1, "healthcare": 2}  
TICKERS = {"tech":["AAPL", "MSFT", "GOOG", "AMZN", "NVDA"],
           "energy":["XOM", "CVX", "COP", "SLB", "ENPH"],
           "healthcare":["JNJ", "PFE", "MRK", "ABT", "AMGN"]
          }

# PARAMETERS (tb include in yml file)
WINDOW_SIZE = 30
BATCH_SIZE = 64
BOTTLENECK_DIM = 16
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10

# Data extraction period
START_DATE = "2020-01-01"
END_DATE = "2025-12-31"

def main():
    
    all_X, all_S = [], []
    for sector in SECTOR_INDEX.keys():
        all_dfs = []
        # Fetch data
        for ticker in TICKERS[sector]:
            df = fetch_and_save_tickers(ticker, START_DATE, END_DATE, DATA_DIR)
            if not df.empty:
                all_dfs.append(df)
    
        if not all_dfs:
            print("No data fetched. Exiting.")
            return
    
        # Merge all tickers on date index
        combined_df = pd.concat(all_dfs, axis=1, join="inner").sort_index()
    
        # Save combined dataset
        combined_file = os.path.join(DATA_DIR, f"combined_stocks_{sector}.csv")
        combined_df.to_csv(combined_file, index=True)
        print(f"Saved combined dataset to {combined_file} with shape {combined_df.shape[0]}")
    
        # Preprocess dataset
        filepath = os.path.join(DATA_DIR, f"combined_stocks_{sector}.csv")
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        X, S, _, _, _ = preprocess_multivariate(df, sector, SECTOR_INDEX, window_size=WINDOW_SIZE)
        all_X.append(X) 
        all_S.append(S) 

    # Concatenate across sectors
    X = np.concatenate(all_X, axis=0) # (total_samples, window, num_features)
    S = np.concatenate(all_S, axis=0) # (total_samples, sector dim)

    # Create torch dataset and data loader
    dataset = TimeSeriesDataset(X, S)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    batch = next(iter(loader))
    print(f"Data batch shape (X): {batch[0].shape}") # (batch size, sequence size (window), features)
    print(f"Data batch shape (S): {batch[1].shape}") # (batch size, feature)

    # training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder(window_size=WINDOW_SIZE, 
                        num_features=len(TICKERS[sector]), 
                        sector_dim=len(SECTOR_INDEX), 
                        bottleneck_dim=BOTTLENECK_DIM).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    for epoch in range(NUM_EPOCHS):
        epoch_losses = []
        for x_batch, s_batch in loader:
            x_batch = x_batch.to(device)
            s_batch = s_batch.to(device)
            
            recon = model(x_batch, s_batch) # forward pass
            loss = criterion(recon, x_batch) # compute loss
            
            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step() 

            epoch_losses.append(loss.item()) # store batch loss

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {avg_loss: .4f}")

if __name__ == "__main__":
    main()
