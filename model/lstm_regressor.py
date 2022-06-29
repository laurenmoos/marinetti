import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import MeanSquaredError


class LSTMRegressor(pl.LightningModule):
    """
    Standard PyTorch Lightning module:
    https://pytorch-lightning.readthedocs.io/en/latest/lightning_module.html
    """

    def __init__(self,
                 n_features,
                 hidden_size,
                 seq_len,
                 batch_size,
                 num_layers,
                 dropout,
                 learning_rate,
                 criterion):
        super(LSTMRegressor, self).__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.criterion = criterion
        self.learning_rate = learning_rate

        self.lstm = nn.LSTM(input_size=n_features,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

        self.train_acc = MeanSquaredError()
        self.valid_acc = MeanSquaredError()

    def forward(self, x):
        # lstm_out = (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)
        y_pred = self.linear(lstm_out[:, -1])
        return y_pred

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        result = self.criterion(y_hat, y)
        self.train_acc(y_hat, y)
        self.log("train_res", self.train_acc, on_step=True, on_epoch=False)
        return result

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        result = self.criterion(y_hat, y)
        self.valid_acc(y_hat, y)
        self.log("valid_res", self.train_acc, on_step=True, on_epoch=False)
        return result

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        result = self.criterion(y_hat, y)
        return result