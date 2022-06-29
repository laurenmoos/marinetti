from particle_accelerator_dataset import ParticleAcceleratorDataModule
from pytorch_lightning.trainer import Trainer, seed_everything
from lstm_regressor import LSTMRegressor
import torch.nn as nn
from argparse import ArgumentParser
from pytorch_lightning.loggers import TensorBoardLogger
import random
import pandas as pd
import numpy as np

from pl_bolts.models.autoencoders import AE



params_dict = {
    'control_params': [
        'Fwd1Amp',
        'SparePhs',
        'LP_Phase',
        'Rev_Phs',
        'Fwd1Phs',
        'CavAmp',
        'SpareAmp',
        'Cavphs',
        'RevAmp',
        'LP_Amp',
        'Fwd2Amp'
    ], 'lcam_rms_cent': [
        'LCamGauss_xCentroid',
        'LCamGauss_yCentroid',
        'LCamGauss_uCentroid',
        'LCamGauss_vCentroid',
        'LCamGauss_xrms',
        'LCamGauss_yrms',
        'LCamGauss_urms',
        'LCamGauss_vrms'
    ], 'scam_rms_cent': [
        'SCam3_Gauss_xCentroid',
        'SCam3_Gauss_yCentroid',
        'SCam3_Gauss_uCentroid',
        'SCam3_Gauss_vCentroid',
        'SCam3_Gauss_xrms',
        'SCam3_Gauss_yrms',
        'SCam3_Gauss_urms',
        'SCam3_Gauss_vrms'
    ]
}


def loop(hparam):

    hash = random.getrandbits(128)
    seed_everything(1)

    exp_name = f"${hash}"
    logger = TensorBoardLogger("logs", name=exp_name)
    print(f" Logging for Experiment {exp_name}")
    trainer = Trainer(
        max_epochs=hparam.max_epochs,
        progress_bar_refresh_rate=2,
        logger=logger)

    x_cols = [
        'LCamGauss_xCentroid',
        'LCamGauss_yCentroid',
        'LCamGauss_uCentroid',
        'LCamGauss_vCentroid',
        'LCamGauss_xrms',
        'LCamGauss_yrms',
        'LCamGauss_urms',
        'LCamGauss_vrms'
    ]
    y_cols = [
        'SCam3_Gauss_xCentroid',
        'SCam3_Gauss_yCentroid',
        'SCam3_Gauss_uCentroid',
        'SCam3_Gauss_vCentroid',
        'SCam3_Gauss_xrms',
        'SCam3_Gauss_yrms',
        'SCam3_Gauss_urms',
        'SCam3_Gauss_vrms'
    ]

    # model = LSTMRegressor(
    #     n_features=hparam.n_features,
    #     hidden_size=hparam.hidden_size,
    #     seq_len=hparam.seq_len,
    #     batch_size=hparam.batch_size,
    #     criterion=hparam.criterion,
    #     num_layers=hparam.num_layers,
    #     dropout=hparam.dropout,
    #     learning_rate=hparam.learning_rate
    # )

    ae = AE(input_height=32)
    model = ae

    ae.freeze()

    dm = ParticleAcceleratorDataModule(
        params_dict['control_params'],
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
    # TODO: this might not be the right loss
    parser.add_argument("--criterion", default=nn.MSELoss())
    parser.add_argument("--max_epochs", default=100)
    parser.add_argument("--n_features", default=11)
    parser.add_argument("--hidden_size", default=16)
    parser.add_argument("--rows_to_drop", default=1000)
    # TODO: this doesn't seem right
    parser.add_argument("--num_layers", default=3)
    parser.add_argument("--dropout", default=0.4)
    parser.add_argument("--learning_rate", default=0.001)
    parser.add_argument("--accelerator", default=None)
    parser.add_argument("--devices", default=None)
    args = parser.parse_args()

    loop(args)
