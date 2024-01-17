""" DataModule for Moseley's harmonic oscillator.
"""
from typing import Callable, List

import pandas as pd
import pytorch_lightning as pl
import torch

from src.ml import train_val_test_split, normalize, get_numeric


class MoseleyPINNDataModule(pl.LightningDataModule):

    TRAIN_RATIO = 0.6
    VAL_RATIO = 0.2
    TEST_RATIO = 0.2

    def __init__(
        self,
        path_to_data: str,
        targets: List[str],
        args: Callable,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(args.__dict__)
        self.save_hyperparameters("targets")
        self.save_hyperparameters("path_to_data")
        
        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None
        self.scaler = None
        self.column_names = None
        self.target_names = None
        self.in_features = None
        self.out_features = None

    @staticmethod
    def load_data(path_to_data) -> pd.DataFrame:
        return pd.read_parquet(path_to_data)

    def prepare_data(self) -> None:
        # https://github.com/Lightning-AI/lightning/issues/11528
        pass

    def setup(self) -> None:
        # Load, create dataset
        df = self.load_data(self.hparams.path_to_data)
        df_numeric = get_numeric(df)

        # No need to shuffle -> DataLoader handles this
        # Split, normalize
        x_train, x_val, x_test, y_train, y_val, y_test = train_val_test_split(
            df=df_numeric,
            ratio=(self.TRAIN_RATIO, self.VAL_RATIO, self.TEST_RATIO),
            target_columns=self.hparams.targets,
        )

        self.column_names = list(x_train.columns)
        self.target_names = list(y_train.columns)
        self.in_features = x_train.shape[1]
        self.out_features = y_train.shape[1]

        x_train_norm, x_val_norm, x_test_norm, self.scaler = normalize(
            x_train,
            x_val,
            x_test,
            norm_type="z-score",
        )

        self.dataset_train = torch.utils.data.TensorDataset(
            torch.Tensor(x_train_norm),
            torch.Tensor(y_train.values),
        )
        self.dataset_val = torch.utils.data.TensorDataset(
            torch.Tensor(x_val_norm),
            torch.Tensor(y_val.values),
        )
        self.dataset_test = torch.utils.data.TensorDataset(
            torch.Tensor(x_test_norm),
            torch.Tensor(y_test.values),
        )
       
    def train_dataloader(self): 
        return torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
        )
    
    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
        )
