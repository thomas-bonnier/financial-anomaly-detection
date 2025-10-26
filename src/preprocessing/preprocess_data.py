import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_multivariate(df: pd.DataFrame, window_size: int = 30):
    """
    Scale multivariate time series and create sliding windows.
    Args: 
        df: DataFrame with dates as index and tickers as columns
        window_size: number of days per window
    Returns:
        X: numpy array of shape (num_samples, window_size, num_features)
        scaler: fitted MinMaxScaler for inverse transforms
    """
    # scale each ticker to [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df.values)

    # create sliding windows
    X = []
    for i in range(len(scaled) - window_size):
        X.append(scaled[i:i+window_size])
    X = np.array(X)

    return X, scaler

    

