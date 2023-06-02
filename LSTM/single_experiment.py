import numpy as np
import torch

from model.volatility_lstm import VolatilityLSTM

from typing import Tuple
from copy import deepcopy


def train_model(model: VolatilityLSTM,
                X_train: torch.Tensor,
                y_train: torch.Tensor,
                X_val: torch.Tensor,
                y_val: torch.Tensor,
                max_iter: int,
                patience: int) -> Tuple[VolatilityLSTM, float, int]:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = torch.nn.MSELoss()

    best_val_loss = np.inf
    final_epoch = None
    best_model = None
    counter = 0

    for epoch in range(max_iter):
        optimizer.zero_grad()
        pred_train, _ = model(X_train)
        loss_train = criterion(pred_train[:, :-1], y_train)
        loss_train.backward()
        optimizer.step()

        pred_val, _ = model(X_val)
        loss_val = criterion(pred_val[:, :-1], y_val)

        if loss_val < best_val_loss:
            best_val_loss = loss_val
            best_model = deepcopy(model)
        else:
            counter += 1

        if counter >= patience:
            final_epoch = epoch
            break

    last_epoch = final_epoch or max_iter

    return best_model, best_val_loss.item(), last_epoch + 1


def test_model(model: VolatilityLSTM,
               X_test: torch.Tensor,
               y_test: torch.Tensor) -> Tuple[float, float]:
    criterion = torch.nn.MSELoss()

    pred_forced, _ = model(X_test)
    pred_infer = model.predict(X_test[:, 0, :].unsqueeze(1), n_days=y_test.size(1) - 1, keep_init=True)

    loss_forced = criterion(pred_forced[:, :-1], y_test)
    loss_infer = criterion(pred_infer, y_test)

    return loss_forced.item(), loss_infer.item()
