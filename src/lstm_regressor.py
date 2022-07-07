import pytorch_lightning as pl
import torch
import torch.nn as nn
from dataset import ParticleAcceleratorDataModule
from pytorch_lightning.trainer import Trainer, seed_everything
from argparse import ArgumentParser
from pytorch_lightning.loggers import TensorBoardLogger
import random
from rmse import RMSELoss, RMSEAcc
from config import params_dict


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
        self.train_acc = RMSEAcc()
        self.valid_acc = RMSEAcc()

    def forward(self, x):
        # lstm_out = (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)
        y_pred = self.linear(lstm_out[:, -1])
        return y_pred

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        print(f"Train x has first element with shape {batch[0].shape}")
        print(f"Train y has first element with shape {batch[1].shape}")
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


def loop(hparam):
    h = random.getrandbits(128)
    seed_everything(1)

    exp_name = f"experiment_{h}"
    logger = TensorBoardLogger(".logs/", name=exp_name)
    print(f" Logging for Experiment {exp_name}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainer = Trainer(max_epochs=hparam.max_epochs, progress_bar_refresh_rate=2, logger=logger)

    x_cols = params_dict['control_params']
    y_cols = params_dict['target']

    model = LSTMRegressor(
        n_features=hparam.n_features,
        hidden_size=hparam.hidden_size,
        seq_len=hparam.seq_len,
        batch_size=hparam.batch_size,
        criterion=hparam.criterion,
        num_layers=hparam.num_layers,
        dropout=hparam.dropout,
        learning_rate=hparam.learning_rate
    )

    dm = ParticleAcceleratorDataModule(
        x_cols,
        y_cols,
        hparam.rows_to_drop,
        seq_len=hparam.seq_len,
        batch_size=hparam.batch_size
    )

    dm.setup()

    trainer.fit(model, dm)
    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seq_len", default=24)
    parser.add_argument("--batch_size", default=70)

    '''
    LSTM Imputation/Forecasting Hyper-Parameters
    '''
    parser.add_argument("--criterion", default=RMSELoss())
    parser.add_argument("--max_epochs", default=100)
    parser.add_argument("--n_features", default=11)
    parser.add_argument("--hidden_size", default=16)
    parser.add_argument("--rows_to_drop", default=1000)
    parser.add_argument("--num_layers", default=3)
    parser.add_argument("--dropout", default=0.4)
    parser.add_argument("--learning_rate", default=0.0001)
    parser.add_argument("--accelerator", default=None)
    parser.add_argument("--devices", default=None)
    args = parser.parse_args()

    loop(args)
