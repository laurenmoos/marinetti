from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

from pytorch_lightning.trainer import Trainer, seed_everything
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.metrics import RMSE
from argparse import ArgumentParser
from pytorch_lightning.loggers import TensorBoardLogger
import random
from config import params_dict
import pandas as pd
from pytorch_forecasting.data.encoders import NaNLabelEncoder, TorchNormalizer
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, RecurrentNetwork
from sklearn.preprocessing._data import RobustScaler
import math


def data(hparam):
    df = pd.read_pickle('df-no-miss.pk')
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

    # want to add robust scaler for features not sure what is going wrong
    training = TimeSeriesDataSet(
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
        time_varying_known_reals=params_dict['control_params']
    )

    validation = TimeSeriesDataSet.from_dataset(training, df, min_prediction_idx=training.index.time.max() + 1,
                                                stop_randomization=True)
    train_dataloader = training.to_dataloader(train=True, batch_size=hparam['batch_size'], num_workers=2)
    val_dataloader = validation.to_dataloader(train=False, batch_size=hparam['batch_size'], num_workers=2)

    return training, validation, train_dataloader, val_dataloader


def loop(hparam):
    h = random.getrandbits(4)
    seed_everything(1)

    exp_name = f"{hparam['model']}_{h}"
    print(f"Logging for experiment {exp_name}")
    # logger = TensorBoardLogger("logs/", name=exp_name)

    training, validation, train_dataloader, val_dataloader = data(hparam)

    # early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-9, patience=10, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()
    trainer = Trainer(
        enable_checkpointing=False,
        max_epochs=hparam['max_epochs'],
        gpus=0,
        gradient_clip_val=hparam["max_grad_norm"],
        callbacks=[lr_logger],
    )

    loss = RMSE()

    if hparam['model'] == 'Transformer':
        # create the model
        m = TemporalFusionTransformer.from_dataset(
            training,
            hidden_size=hparam['hidden_size'],
            attention_head_size=hparam['attention_head_size'],
            dropout=hparam['dropout'],
            hidden_continuous_size=hparam["hidden_continuous_size"],
            # note: output size is for the number of quintiles for the criterion
            output_size=1,
            loss=QuantileLoss(),
            log_interval=hparam['log_interval'],
            logging_metrics=RMSE(),
            reduce_on_plateau_patience=hparam['reduce_on_plateau_patience']
        )
    else:
        m = RecurrentNetwork.from_dataset(
            training,
            hidden_size=hparam['hidden_size'],
            loss=loss,
            rnn_layers=hparam['rnn_layers'],
            dropout=hparam['dropout']
        )
    # find optimal learning rate (set limit_train_batches to 1.0 and log_interval = -1)
    res = trainer.tuner.lr_find(
        m, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, early_stop_threshold=1000.0,
        max_lr=hparam['max_learning_rate'],
    )

    fig = res.plot(show=True, suggest=True)
    fig.show()

    # fit the model
    trainer.fit(m, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, )


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--model", choices=['Transformer', 'LSTM'], default='LSTM')

    parser.add_argument("--accelerator", default=None)
    parser.add_argument("--devices", default=None)
    parser.add_argument("--log-interval", default=2)

    parser.add_argument("--batch_size", default=70)
    parser.add_argument("--max_grad_norm", default=0.1)
    parser.add_argument("--criterion", choices=['QuantileLoss', 'RMSELoss'], default='RMSELoss')
    parser.add_argument("--max_epochs", default=100)
    parser.add_argument("--reduce_on_plateau_patience", default=100)
    parser.add_argument("--dropout", default=0.4)
    parser.add_argument("--max_learning_rate", default=0.01)

    '''
    Task Configuration
    '''
    parser.add_argument("--predictor_length", default=1)
    parser.add_argument("--encoder_length", default=12)

    '''
    LSTM HyperParameters
    '''
    parser.add_argument("--hidden_size", default=16)
    parser.add_argument("--rnn_layers", default=12)

    '''
    Transformer HyperParameters
    '''
    parser.add_argument("--attention_head_size", default=1)
    parser.add_argument("--hidden_continuous_size", default=16)
    parser.add_argument("--output_size", default=7)

    args = vars(parser.parse_args())

    loop(args)
