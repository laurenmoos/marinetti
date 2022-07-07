import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader

import torch
import numpy as np
from torch.utils.data import Dataset


class TimeseriesDataset(Dataset):
    """
    Custom Dataset subclass.
    Serves as input to DataLoader to transform X
      into sequence data using rolling window.
    DataLoader using this dataset will output batches
      of `(batch_size, seq_len, n_features)` shape.
    Suitable as an input to RNNs.
    """

    def __init__(self, X: np.ndarray, seq_len: int = 1):
        self.X = torch.tensor(X).float()
        self.seq_len = seq_len

    def __len__(self):
        return self.X.__len__() - (self.seq_len - 1)

    def __getitem__(self, index):
        return self.X[index:index + self.seq_len]


class AutoencoderDataModule(pl.LightningDataModule):
    """
    PyTorch Lighting DataModule subclass:
    https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html

    Serves the purpose of aggregating all data loading
      and processing work in one place.
    """

    def __init__(self, xcols, to_drop, seq_len=1, batch_size=128, num_workers=0):
        super().__init__()
        self.xcols = xcols

        self.to_drop = to_drop

        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.columns = None
        self.preprocessing = None

    def prepare_data(self):
        pass

    def setup(self, stage=None):

        df = pd.read_pickle('pickled_df_ALL.pk').diff()

        remove_n = self.to_drop
        drop_indices = np.random.choice(df.index, remove_n, replace=False)
        df = df.drop(drop_indices)

        X = df[self.xcols].dropna()

        X_cv, X_test = train_test_split(X, test_size=0.2, shuffle=False)

        X_train, X_val = train_test_split(X_cv, test_size=0.25, shuffle=False)
        # TODO: change this
        preprocessing = RobustScaler()
        preprocessing.fit(X_train)

        if stage == 'fit' or stage is None:
            self.X_train = preprocessing.transform(X_train)
            self.X_val = preprocessing.transform(X_val)

        if stage == 'test' or stage is None:
            self.X_test = preprocessing.transform(X_test)

    def train_dataloader(self):
        train_dataset = TimeseriesDataset(self.X_train, seq_len=self.seq_len)
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=False,
                                  num_workers=self.num_workers)

        return train_loader

    def val_dataloader(self):
        val_dataset = TimeseriesDataset(self.X_val, seq_len=self.seq_len)
        val_loader = DataLoader(val_dataset,
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=self.num_workers)

        return val_loader

    def test_dataloader(self):
        test_dataset = TimeseriesDataset(self.X_test, seq_len=self.seq_len)
        test_loader = DataLoader(test_dataset,
                                 batch_size=self.batch_size,
                                 shuffle=False,
                                 num_workers=self.num_workers)

        return test_loader
