import torch
import torch.nn as nn


class VolatilityLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        sigma_series = nn.functional.sigmoid(preds)
        loss = 2 * sigma_series.log10() + (targets / sigma_series).pow(2)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise NotImplementedError
