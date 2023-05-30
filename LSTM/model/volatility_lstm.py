import torch
import torch.nn as nn
from typing import List, Tuple
from model.tensor_funcs import tensor_window_slide


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
                self.dense_layers = init_dense_layer + [
                    nn.Linear(in_features=dense_hidden_sizes, out_features=dense_hidden_sizes) for _ in
                    range(n_dense_layers - 1)]
                self.fc = nn.Linear(in_features=dense_hidden_sizes, out_features=1)

            elif isinstance(dense_hidden_sizes, list):
                if len(dense_hidden_sizes) != n_dense_layers:
                    raise IndexError(f'Number of hidden sizes should match the number of layers\n'
                                     f'Got {len(dense_hidden_sizes)} and {n_dense_layers}')

                dense_hidden_sizes = [lstm_hidden_size] + dense_hidden_sizes
                self.dense_layers = [
                    nn.Linear(in_features=dense_hidden_sizes[i], out_features=dense_hidden_sizes[i + 1]) for i in
                    range(n_dense_layers)]
                self.fc = nn.Linear(in_features=dense_hidden_sizes[-1], out_features=1)

        else:
            self.dense_layers = []
            self.fc = nn.Linear(in_features=lstm_hidden_size, out_features=1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes forward pass
        :param X: tensor of shape (batch size, sequence length, window_size)
        :return: tensor of shape (batch_size, sequence length)
        """
        y, hiddens = self.lstm(X)
        for layer in self.dense_layers:
            y = self.relu(self.dropout(layer(y)))

        y = self.fc(y)

        return y.squeeze(), hiddens

    def predict(self, X: torch.Tensor, n_days: int, keep_init: bool) -> torch.Tensor:
        """
        predicts the volatility from a sequence of data
        :param X: tensor of shape (batch size, sequence_length, window_size)
        :param n_days: number of days to predict in the future
        :param keep_init: if True, appends its prediction for the given days to the output
        :return: tensor of shape (batch size, n_days) if keep_init == False.
                 otherwise (batch size, sequence length + n_days)
        """
        init_preds, hiddens = self.forward(X)
        final_pred = init_preds[:, -1]

        new_preds = torch.zeros(X.size(0), n_days).to(device=X.device)
        new_preds[:, 0] = final_pred

        # slides the window one day to the right
        next_input = tensor_window_slide(X[:, -1, :].view(X.size(0), 1, X.size(-1)), final_pred)

        for i in range(1, n_days):
            next_pred, hiddens = self.forward(next_input)
            new_preds[:, i] = next_pred
            next_input = tensor_window_slide(next_input, next_pred)

        if keep_init:
            return torch.cat((init_preds, new_preds), dim=1)
        else:
            return new_preds

    def dense_cuda(self):
        for layer in self.dense_layers:
            layer = layer.to(device='cuda')
