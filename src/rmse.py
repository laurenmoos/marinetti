import torch
from torch import nn
from torchmetrics import Metric


# create a nn class (just-for-fun choice :-)
class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


class RMSEAcc(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("acc", default=torch.DoubleTensor(0), dist_reduce_fx="sum")

        self.mse = nn.MSELoss()

    def update(self, y_preds: torch.Tensor, y: torch.Tensor):
        y_preds, y = self._input_format(y_preds, y)
        self.acc = torch.sqrt(self.mse(y_preds, y))

    def _input_format(self, y_preds, y):
        return y_preds, y

    def compute(self):
        return self.acc
