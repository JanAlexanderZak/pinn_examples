""" DataModule for thick-walled cylinder (Lame problem).
"""
import os

from typing import Callable, Tuple

import numpy as np
import pytorch_lightning as pl
import torch


class ConcatDatasets(torch.utils.data.Dataset):
    """ ConcatDatasets joins different datasets (e.g. collocation points, BC)
        into a tuple and provides the functionalities needed for training_step.
    """
    def __init__(self, *datasets) -> None:
        self.datasets = datasets

    def __getitem__(self, idx: int) -> Tuple[torch.utils.data.dataset.TensorDataset]:
        return tuple(self.datasets[0][i][idx] for i in range(len(self.datasets[0])))

    def __len__(self) -> int:
        return min(len(dataset) for dataset in self.datasets[0])


class CylinderPINNDataModule(pl.LightningDataModule):
    def __init__(
        self,
        path_to_data: str,
        args: Callable,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(args.__dict__)
        self.save_hyperparameters("path_to_data")

        self.dataset_train = None
        self.dataset_train_BC = None
        self.dataset_test = None
        self.column_names = None
        self.target_names = None
        self.in_features = None
        self.out_features = None

    @staticmethod
    def load_data(path_to_data) -> tuple:
        return (
            np.load(os.path.join(path_to_data, "r_train.npy")),
            np.load(os.path.join(path_to_data, "x_train_BC.npy")),
            np.load(os.path.join(path_to_data, "y_train_BC.npy")),
            np.load(os.path.join(path_to_data, "r_star.npy")),
        )

    def prepare_data(self) -> None:
        pass

    def setup(self) -> None:
        r_train, x_train_BC, y_train_BC, r_star = (
            self.load_data(self.hparams.path_to_data)
        )

        self.column_names = list(["r"])
        self.target_names = list(["u"])
        self.in_features = 1   # model input: radial coordinate r
        self.out_features = 1  # model output: radial displacement u

        self.dataset_train = torch.utils.data.TensorDataset(
            torch.Tensor(r_train),
        )

        self.dataset_train_BC = torch.utils.data.TensorDataset(
            torch.Tensor(x_train_BC),
            torch.Tensor(y_train_BC),
        )

        self.dataset_test = torch.utils.data.TensorDataset(
            torch.Tensor(r_star),
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
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.persistent_workers,
            shuffle=False,
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_test,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.persistent_workers,
            shuffle=False,
        )
