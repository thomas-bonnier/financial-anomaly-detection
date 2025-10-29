import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_multivariate(df: pd.DataFrame, sector_name: str, 
                            sector_index: dict, window_size: int = 30):
    """
    Scale multivariate log returns, create sliding windows and add sector information.
    Args: 
        df: DataFrame with dates as index and tickers as columns (log returns)
        sector_index: dictionary of sector names (keys) and indices
        sector_name: sector name (ex. "tech")
        window_size: number of days per window
    Returns:
        X: numpy array of shape (num_samples, window_size, num_features)
        scaler: fitted MinMaxScaler for inverse transforms
    """
    # compute log returns
    log_returns = np.log(df / df.shift(1)).dropna()
    
    # scale each ticker’s returns to [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(log_returns.values)

    # create sliding windows
    X, dates = [], []
    for i in range(len(scaled) - window_size + 1):
        X.append(scaled[i:i+window_size])
        # single timestamp (last date in the window)
        dates.append(log_returns.index[i+window_size - 1])
    X = np.array(X)

    # create sector one-hot per window
    s_vec = np.zeros(len(sector_index), dtype = float)
    s_vec[sector_index[sector_name]] = 1.0
    S = np.tile(s_vec, (X.shape[0], 1)) # (num_samples, feature sector_dim)
    
    return X, S, scaler, log_returns, dates

    

