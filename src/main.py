from pytorch_lightning.callbacks import LearningRateMonitor

from pytorch_lightning.trainer import Trainer, seed_everything
from pytorch_forecasting.metrics import QuantileLoss, RMSE
from argparse import ArgumentParser
import random
from config import params_dict
import pandas as pd
from pytorch_forecasting.data.encoders import NaNLabelEncoder, TorchNormalizer
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, RecurrentNetwork
from sklearn.preprocessing._data import RobustScaler
import math

from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

DATA_LOC = 'df-no-miss.pk'


def data(hparam):
    df = pd.read_pickle(DATA_LOC)
    df.reset_index(inplace=True)

    # diff the data
    df = df.diff().dropna()
    N = len(df.index)

    # create monotonic integer based time column
    df['Time'] = pd.Series([int(x) for x in range(N + 1)])
    # df['Time'] = df['Time'].map(int)

    # dummy variable to allow for construction of the time series dataset
    df['Group'] = pd.Series(['1.0' for x in range(N + 1)], dtype="category")

    # define dataset
    max_prediction_length = hparam['predictor_length']
    max_encoder_length = hparam['encoder_length']

    encoder_res = {val: RobustScaler() for val in params_dict['control_params']}

    training_cutoff = df.iloc[math.floor(N * 0.8)]['Time']

    # want to add robust scaler for features not sure what is going wrong
    training = TimeSeriesDataSet(
        df[lambda x: x.Time < training_cutoff],
        target='SCam3_COM',
        time_idx='Time',
        group_ids=['Group'],
        categorical_encoders={'Group': NaNLabelEncoder(), 'Time': NaNLabelEncoder()},
        scalers=encoder_res,
        static_categoricals=["Group"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_unknown_reals=['SCam3_COM'],
        time_varying_known_reals=params_dict['control_params'] + ['LCam1COM']
    )

    # TODO: think about contents of validation and test set, for validation maybe do cross-fold
    # for test back-test it on the entire observed dataset - not perfect but ok until get more data
    validation = TimeSeriesDataSet.from_dataset(training, df,
                                                min_prediction_idx=training_cutoff + 1,
                                                stop_randomization=True)

    test = TimeSeriesDataSet(
        df,
        target='SCam3_COM',
        time_idx='Time',
        group_ids=['Group'],
        categorical_encoders={'Group': NaNLabelEncoder(), 'Time': NaNLabelEncoder()},
        scalers=encoder_res,
        static_categoricals=["Group"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_unknown_reals=['SCam3_COM'],
        time_varying_known_reals=params_dict['control_params'] + ['LCam1COM']
    )

    return training, validation, test


def loop(hparam):
    h = random.getrandbits(32)
    seed_everything(1)

    exp_name = f"{hparam['model']}_{h}"
    print(f"Logging for experiment {exp_name}")
    logger = TensorBoardLogger("logs", name=exp_name)

    training, validation, test = data(hparam)

    train_dataloader = training.to_dataloader(train=True, batch_size=hparam['batch_size'], num_workers=8)
    val_dataloader = validation.to_dataloader(train=False, batch_size=hparam['batch_size'], num_workers=8)
    test_dataloader = test.to_dataloader(train=False, batch_size=hparam['batch_size'], num_workers=8)

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=1, verbose=False, mode="min")
    lr_logger = LearningRateMonitor(logging_interval="epoch")
    trainer = Trainer(
        fast_dev_run=False,
        logger=logger,
        max_epochs=hparam['max_epochs'],
        gpus=0,
        gradient_clip_val=hparam["max_grad_norm"],
        callbacks=[lr_logger, early_stop_callback],
    )

    if hparam['model'] == 'Transformer':
        # create the model
        m = TemporalFusionTransformer.from_dataset(
            training,
            hidden_size=hparam['hidden_size'],
            attention_head_size=hparam['attention_head_size'],
            dropout=hparam['dropout'],
            hidden_continuous_size=hparam["hidden_continuous_size"],
            # note: output size is for the number of quintiles for the criterion
            output_size=7,
            loss=QuantileLoss(),
            log_interval=hparam['log_interval'],
            reduce_on_plateau_patience=hparam['reduce_on_plateau_patience']
        )
    else:
        m = RecurrentNetwork.from_dataset(
            training,
            learning_rate=hparam['learning_rate'],
            hidden_size=hparam['hidden_size'],
            log_interval=hparam['log_interval'],
            loss=RMSE(),
            rnn_layers=hparam['rnn_layers'],
            dropout=hparam['dropout'],
            reduce_on_plateau_patience = hparam['reduce_on_plateau_patience']
        )
    # find optimal learning rate (set limit_train_batches to 1.0 and log_interval = -1)
    res = trainer.tuner.lr_find(
        m, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader,
        early_stop_threshold=hparam['early_stop_threshold'],
        max_lr=hparam['max_learning_rate'],
    )

    # fit the model
    trainer.fit(m, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, )

    m.eval(training.d

    # convert the diff to predictions and plot

    # some kind of back-testing logic against a baseline model


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--model", choices=['Transformer', 'LSTM'], default='LSTM')

    parser.add_argument("--accelerator", default=None)
    parser.add_argument("--devices", default=None)
    parser.add_argument("--log-interval", default=2)

    parser.add_argument("--batch_size", default=64)
    parser.add_argument("--max_grad_norm", default=0.1)
    parser.add_argument("--criterion", choices=['QuantileLoss', 'RMSELoss'], default='RMSELoss')
    parser.add_argument("--max_epochs", default=1)
    parser.add_argument("--reduce_on_plateau_patience", default=4)
    parser.add_argument("--dropout", default=0.1)
    '''
    Learning Rate Stuff 
    '''
    parser.add_argument("--early_stop_threshold", default=1000)
    parser.add_argument("--max_learning_rate", default=0.01)
    parser.add_argument("--learning_rate", default=0.009)

    '''
    Task Configuration
    '''
    parser.add_argument("--predictor_length", default=1)
    parser.add_argument("--encoder_length", default=64)

    '''
    LSTM HyperParameters
    '''
    parser.add_argument("--hidden_size", default=64)
    parser.add_argument("--rnn_layers", default=3)

    '''
    Transformer HyperParameters
    '''
    parser.add_argument("--attention_head_size", default=1)
    parser.add_argument("--hidden_continuous_size", default=16)
    parser.add_argument("--output_size", default=7)

    args = vars(parser.parse_args())

    loop(args)
