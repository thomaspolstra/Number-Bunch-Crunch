from torch.utils.data import Dataset, DataLoader
import torch
from pandas import DataFrame
from typing import Tuple
import numpy as np


class TickerDataset(Dataset):
    def __init__(self, ticker_df: DataFrame, window_size: int, mean: float, std: float):
        super().__init__()
        self.tickers = ticker_df['Adj Close'].columns.values
        self.ticker_data = ticker_df
        self.window_size = window_size
        self.mean = mean
        self.std = std

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """returns input and target tensors for one ticker symbol"""
        daily_close = self.ticker_data['Adj Close']

        log_returns = torch.tensor(daily_close[self.tickers[index]], dtype=torch.float).log10().diff()
        log_returns = (log_returns - self.mean) / self.std
        input = log_returns.unfold(0, self.window_size, 1)  # turns 1d tensor into 2d of shape
                                                            # (n_days - window_size, window_size)
        target = log_returns[self.window_size:]  # 1d tensor of shape (n_days - window_size)

        return input, target

    def __len__(self):
        return len(self.tickers)


def create_dataloaders(ticker_df: DataFrame,
                       window_size: int,
                       batch_size: int,
                       shuffle: bool,
                       n_workers: int,
                       val_ratio: float,
                       test_ratio: float) -> Tuple[Tuple[DataLoader, DataLoader, DataLoader], float, float]:
    """
    :param ticker_df: DataFrame for tickers
    :param window_size: number of days in rolling window
    :param batch_size: batch size
    :param shuffle: if the tickers will be shuffled (not the dates)
    :param n_workers: number of workers for the data loader
    :param val_ratio: the ratio of days for the validation set
    :param test_ratio: the ratio of days for the test set
    :return: three data loaders (train, val, test)
    """
    dataframes = train_val_test_split(ticker_df, val_ratio, test_ratio)
    mean, std = find_mean_std(dataframes[0])
    shuffles = (shuffle, False, False)
    loaders = []

    for df, should_shuffle in zip(dataframes, shuffles):
        loaders.append(DataLoader(TickerDataset(df, window_size, mean, std),
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=n_workers))

    return tuple(loaders), mean, std


def train_val_test_split(data: DataFrame,
                         val_ratio: float,
                         test_ratio: float) -> Tuple[DataFrame, DataFrame, DataFrame]:
    n_days = len(data)
    n_days_val = int(n_days * val_ratio)
    n_days_test = int(n_days * test_ratio)

    train_data = data[:-n_days_val - n_days_test].copy()
    val_data = data[-n_days_val - n_days_test:-n_days_test].copy()
    test_data = data[-n_days_test:].copy()

    return train_data, val_data, test_data


def find_mean_std(data: DataFrame) -> Tuple[float, float]:
    log_closes = data['Adj Close'].apply(lambda x: np.log10(x)).diff().dropna()
    mean = log_closes.mean().mean()
    std = log_closes.values.std(ddof=1)
    return mean, std


def find_min_max(data: DataFrame) -> Tuple[float, float]:
    log_closes = data['Adj Close'].apply(lambda x: np.log10(x)).diff().dropna()
    min = log_closes.min().min()
    max = log_closes.max().max()
    return min, max
