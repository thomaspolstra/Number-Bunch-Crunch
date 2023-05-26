from torch.utils.data import Dataset, DataLoader
import torch
from download_or_read_data import download_or_read_data
from typing import List, Tuple


class TickerDataset(Dataset):
    def __init__(self, tickers: List[str], start_date: str, end_date: str, window_size: int):
        super().__init__()
        self.tickers = tickers
        self.ticker_data = download_or_read_data(tickers, start_date, end_date).dropna()
        self.window_size = window_size

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """returns input and target tensors for one ticker symbol"""
        daily_close = self.ticker_data['Adj Close']
        pct_changes = daily_close.pct_change() * 100

        pct_changes_tensor = torch.tensor(pct_changes[self.tickers[index]], dtype=torch.float)
        input = pct_changes_tensor.unfold(0, self.window_size, 1)  # turns 1d tensor into 2d of shape
                                                                   # (n_days - window_size, window_size)
        target = pct_changes_tensor[self.window_size:]  # 1d tensor of shape (n_days - window_size)

        return input, target

    def __len__(self):
        return len(self.tickers)


def create_dataloader(tickers: List[str],
                      start_date: str,
                      end_date: str,
                      window_size: int,
                      batch_size: int,
                      shuffle: bool) -> DataLoader:
    dataset = TickerDataset(tickers, start_date, end_date, window_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
