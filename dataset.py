import torch
from torch.utils.data import Dataset
import pandas as pd


class EntropyDataset(Dataset):
    """
    Drop-in Dataset for entropy / macro time-series models.

    Assumes:
    - Input data is a CSV or DataFrame
    - Target column exists
    - Optional rolling window (sequence length)
    """

    def __init__(
            self,
            data,
            feature_cols,
            target_col,
            window_size=1,
            device="cpu",
            dtype=torch.float32,
    ):
        """
        Parameters
        ----------
        data : str | pd.DataFrame
            Path to CSV or already-loaded DataFrame
        feature_cols : list[str]
            Feature column names
        target_col : str
            Target column name
        window_size : int
            Number of past steps per sample
        device : str
            cpu or cuda
        dtype : torch.dtype
        """

        if isinstance(data, str):
            self.df = pd.read_csv(data)
        else:
            self.df = data.copy()

        self.feature_cols = feature_cols
        self.target_col = target_col
        self.window_size = window_size
        self.device = device
        self.dtype = dtype

        self._prepare_arrays()

    def _prepare_arrays(self):
        X = self.df[self.feature_cols].values
        y = self.df[self.target_col].values

        self.X = torch.tensor(X, dtype=self.dtype)
        self.y = torch.tensor(y, dtype=self.dtype)

    def __len__(self):
        return len(self.df) - self.window_size + 1

    def __getitem__(self, idx):
        """
        Returns:
        --------
        X : (window_size, num_features)
        y : scalar or (1,)
        """

        x_window = self.X[idx : idx + self.window_size]
        y_target = self.y[idx + self.window_size - 1]

        return (
            x_window.to(self.device),
            y_target.to(self.device),
        )
