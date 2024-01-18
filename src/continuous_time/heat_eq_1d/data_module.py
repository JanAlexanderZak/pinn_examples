""" DataModule for the 1D heat equation.
"""
import os

from typing import Callable, List, Tuple

import pandas as pd
import numpy as np
import pytorch_lightning as pl
import torch


class ConcatDatasets(torch.utils.data.Dataset):
    """ ConcatDatasets joins different datasets (e.g. collocation points, IC, BC)
        into a tuple and provides the functionalities needed for training_step.
        Performance is better than dict-return of dataset.

    References:
        https://pytorch-lightning.readthedocs.io/en/1.0.8/multiple_loaders.html
        https://medium.com/mlearning-ai/manipulating-pytorch-datasets-c58487ab113f
        https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
        https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#ConcatDataset
        https://discuss.pytorch.org/t/how-does-concatdataset-work/60083
    """
    def __init__(self, *datasets) -> None:
        """
        Args:
            datasets (List[Datasets]): list of datasets.
        """
        # datasets[0] cancles outer tuple
        self.datasets = datasets

    def __getitem__(self, idx: int) -> Tuple[torch.utils.data.dataset.TensorDataset]:
        """ Returns each dataset separately within a tuple for access in training_step.

        Args:
            idx (int): batch idx

        Returns:
            Tuple[torch.utils.data.dataset.TensorDataset]: Tuple of datasets to for training_step.
        """
        return tuple(self.datasets[0][i][idx] for i in range(len(self.datasets[0])))

    def __len__(self) -> int:
        """ The total dataset length to determine batch per dataset e.g. 15000/512=30 steps

        Returns:
            int: total dataset lentght
        """
        return min(len(dataset) for dataset in self.datasets[0])


class HeatEq1DPINNDataModule(pl.LightningDataModule):
    def __init__(
        self,
        path_to_data: str,
        args: Callable,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(args.__dict__)
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
        return (
            np.load(os.path.join(path_to_data, "x_train.npy")),
            np.load(os.path.join(path_to_data, "y_train.npy")),
            np.load(os.path.join(path_to_data, "x_train_BC.npy")),
            np.load(os.path.join(path_to_data, "y_train_BC.npy")),
            np.load(os.path.join(path_to_data, "X_star.npy")),
        )
    
    def prepare_data(self) -> None:
        # https://github.com/Lightning-AI/lightning/issues/11528
        pass

    def setup(self) -> None:
        # Load, create dataset
        x_train, _, x_train_BC, y_train_BC, X_star = self.load_data(self.hparams.path_to_data)

        self.column_names = list(["x", "t"])
        self.target_names = list(["y"])
        self.in_features = x_train.shape[1]
        self.out_features = y_train_BC.shape[1]

        self.dataset_train = torch.utils.data.TensorDataset(
            torch.Tensor(x_train),
        )

        self.dataset_train_BC = torch.utils.data.TensorDataset(
            torch.Tensor(x_train_BC),
            torch.Tensor(y_train_BC),
        )

        self.dataset_test = torch.utils.data.TensorDataset(
            torch.Tensor(X_star),
        )
       
    def train_dataloader(self): 
        return torch.utils.data.DataLoader(
            ConcatDatasets(
                [
                    self.dataset_train,
                    self.dataset_train_BC,
                ],
            ),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.persistent_workers,
            shuffle=True,
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            ConcatDatasets(
                [
                    self.dataset_train,
                    self.dataset_train_BC,
                ],
            ),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.persistent_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_test,
            #batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.persistent_workers,
            shuffle=False,
        )
    
    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_test,
            #batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.persistent_workers,
            shuffle=False,
        )
