import torch
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    """
    Create torch dataset
    """
    def __init__(self, X, S):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.S = torch.tensor(S, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
            return self.X[idx], self.S[idx]
        


    

