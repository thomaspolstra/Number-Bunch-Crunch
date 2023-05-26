import torch
import torch.nn as nn
from typing import List


class VolatilityLSTM(nn.Module):
    def __init__(self,
                 window_size: int,
                 lstm_hidden_size: int,
                 n_lstm_layers: int,
                 dense_hidden_sizes: int | List[int],
                 n_dense_layers: int,
                 dropout: float,
                 dropout_lstm: bool):
        """
        :param window_size: size of sliding window
        :param lstm_hidden_size: number of hidden units in lstm
        :param n_lstm_layers: number of stacked lstm layers
        :param dense_hidden_sizes: number of hidden units in each dense layer
                                   if int, all layers have the same number of units
                                   if list, the number of hidden units will follow the list
        :param n_dense_layers: number of dense layers in addition to the final fully connected layer
        :param dropout: dropout percentage. Should be between 0.0 and 1.0
        :param dropout_lstm: if True, applies dropout to the LSTM layer. Otherwise just the dense layers.
        """
        super().__init__()

        if dropout_lstm:
            self.lstm = nn.LSTM(input_size=window_size,
                                hidden_size=lstm_hidden_size,
                                num_layers=n_lstm_layers,
                                dropout=dropout,
                                batch_first=True)
        else:
            self.lstm = nn.LSTM(input_size=window_size,
                                hidden_size=lstm_hidden_size,
                                num_layers=n_lstm_layers,
                                batch_first=True)

        if n_dense_layers > 0:
            if isinstance(dense_hidden_sizes, int):
                init_dense_layer = [nn.Linear(in_features=lstm_hidden_size, out_features=dense_hidden_sizes)]
                self.dense_layers = init_dense_layer + [nn.Linear(in_features=dense_hidden_sizes, out_features=dense_hidden_sizes) for _ in range(n_dense_layers-1)]
                self.fc = nn.Linear(in_features=dense_hidden_sizes, out_features=1)

            elif isinstance(dense_hidden_sizes, list):
                if len(dense_hidden_sizes) != n_dense_layers:
                    raise Exception(f'Number of hidden sizes should match the number of layers\nGot {len(dense_hidden_sizes)} and {n_dense_layers}')
                dense_hidden_sizes = [lstm_hidden_size] + dense_hidden_sizes
                self.dense_layers = [nn.Linear(in_features=dense_hidden_sizes[i], out_features=dense_hidden_sizes[i+1]) for i in range(n_dense_layers)]
                self.fc = nn.Linear(in_features=dense_hidden_sizes[-1], out_features=1)

        else:
            self.dense_layers = []
            self.fc = nn.Linear(in_features=lstm_hidden_size, out_features=1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
