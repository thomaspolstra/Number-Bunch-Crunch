import yfinance as yf
import pandas as pd
from pathlib import Path
import os
from typing import List


def download_or_read_data(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    :param tickers: List of ticker symbols as strings
    :param start_date: start date of data YYYY-MM-DD
    :param end_date: end date of data YYYY-MM-DD
    :return: pandas DataFrame from yfinance

    This automatically save the data in the data/ directory if it doesn't exist yet.
    Otherwise, it just reads already saved data.
    """
    path_name = f'../data/tickers_data_{len(tickers)}_{start_date.replace("-", "_")}_{end_date.replace("-", "_")}.pkl'

    if Path(path_name).is_file():
        return pd.read_pickle(path_name)

    data = yf.download(tickers, start=start_date, end=end_date)
    os.makedirs('../data/', exist_ok=True)

    data.to_pickle(path_name)
    return data
