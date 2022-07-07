import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.nn import functional as tf
from autoencoder_dataset import AutoencoderDataModule
from pytorch_lightning.trainer import Trainer, seed_everything
from argparse import ArgumentParser
from pytorch_lightning.loggers import TensorBoardLogger
import random
from config import params_dict
import pytorch_lightning as pl

from networks import AttnEncoder, AttnDecoder, Encoder, Decoder


class AutoEncForecast(pl.LightningModule):
    def __init__(self, config, input_size):
        """
        Initialize the network.

        Args:
            config:
            input_size: (int): size of the input
        """
        super(AutoEncForecast, self).__init__()
        self.encoder = AttnEncoder(config, config['n_features'])
        self.decoder = AttnDecoder(config)

        self.lr = config['learning_rate']

        # loss terms
        self.reg_factor1 = config['reg_factor1']
        self.reg_factor2 = config['reg_factor2']
        self.max_grad_norm = config['max_grad_norm']
        self.grad_acc = config['gradient_accumulation_steps']

    def regularize_loss(self, loss):
        params = torch.cat([p.view(-1) for name, p in self.named_parameters() if 'bias' not in name])
        loss += self.reg_factor1 * torch.norm(params, 1)

        params = torch.cat([p.view(-1) for name, p in self.named_parameters() if 'bias' not in name])
        loss += self.reg_factor2 * torch.norm(params, 2)

        if self.grad_acc > 1:
            loss = loss / self.grad_acc

        return loss

    def forward(self, encoder_input: torch.Tensor, return_attention: bool = False):
        encoder_output, _ = self.encoder(encoder_input)

        outputs, _ = self.decoder(encoder_output)

        return outputs

    def step(self, batch, batch_idx):
        x_hat = self(batch, return_attention=False)

        recon_loss = F.mse_loss(x_hat, batch, reduction="mean")

        loss = self.regularize_loss(recon_loss)

        logs = {
            "recon_loss": recon_loss,
            "loss": loss,
        }
        return loss, logs

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x_hat = self(batch, return_attention=False)

        recon_loss = F.mse_loss(x_hat, batch, reduction="mean")

        loss = self.regularize_loss(recon_loss)
        return loss

    def test_step(self, batch, batch_idx):
        x_hat = self(batch, return_attention=False)

        recon_loss = F.mse_loss(x_hat, batch, reduction="mean")

        loss = self.regularize_loss(recon_loss)
        self.log("test_loss", loss, on_step=True, on_epoch=False)
        return loss


def loop(hparam):
    h = random.getrandbits(128)
    seed_everything(1)

    exp_name = f"experiment_{h}"
    logger = TensorBoardLogger("model/logs", name=exp_name)
    print(f" Logging for Experiment {exp_name}")

    trainer = Trainer(max_epochs=hparam['max_epochs'], progress_bar_refresh_rate=2, logger=logger,
                      gradient_clip_val=hparam["max_grad_norm"])

    x_cols = params_dict['control_params']

    feature_extraction_model = AutoEncForecast(hparam, 11)

    dm = AutoencoderDataModule(
        x_cols,
        to_drop=0,
        seq_len=hparam['seq_len'],
        batch_size=hparam['batch_size']
    )

    dm.setup()

    torch.set_grad_enabled(True)

    trainer.fit(feature_extraction_model, dm)
    trainer.test(feature_extraction_model, datamodule=dm)



    #combine the resultant vectors (average)
    #concatenate the new context to each timestep
    #feed to the lstm regressor


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seq_len", default=24)
    parser.add_argument("--batch_size", default=70)
    parser.add_argument("--n_features", default=11)

    '''
    Autoencoder Hyper-Parameters 
    '''
    parser.add_argument("--max_epochs", default=100)
    parser.add_argument("--input_att", default=False)
    parser.add_argument("--temporal_att", default=False)
    parser.add_argument("--h_size_encoder", default=16)
    parser.add_argument("--h_size_decoder", default=11)
    parser.add_argument("--reg_factor1", default=False)
    parser.add_argument("--reg_factor2", default=False)
    parser.add_argument("--max_grad_norm", default=False)
    parser.add_argument("--gradient_accumulation_steps", default=False)
    parser.add_argument("--learning_rate", default=0.001)
    parser.add_argument("--denoising", default=False)
    parser.add_argument("--bottleneck", default=8)

    args = vars(parser.parse_args())
    print(args)

    loop(args)
