from typing import Callable

import pandas as pd
import pytorch_lightning as pl
import numpy as np
import torch

from typing import Tuple, List

from src.ml import train_val_test_split, normalize, get_numeric


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
            int: total dataset lenght
        """
        return min(len(dataset) for dataset in self.datasets[0])


class PINNDataModule(pl.LightningDataModule):
    """ The DataModule handles the preprocessing of collocation, IC and BC.
        It yields dataloaders w.r.t. each.
    """
    TRAIN_RATIO = 0.6
    VAL_RATIO = 0.2
    TEST_RATIO = 0.2

    def __init__(
        self,
        path_to_data: str,
        path_to_data_IC: str,
        path_to_data_BC: str,
        targets: List[str],
        args: Callable,
        #dataset_version: str,
    ) -> None:
        """ Initialize and write to self.hyparams.

        Args:
            path_to_data (str): _description_
            path_to_data_IC (str): _description_
            path_to_data_BC (str): _description_
            targets (str): _description_
            args (Callable): args from dl.DeepLearningArguments
        """
        super().__init__()

        self.save_hyperparameters(args.__dict__)
        self.save_hyperparameters("targets")
        self.save_hyperparameters("path_to_data")
        self.save_hyperparameters("path_to_data_IC")
        self.save_hyperparameters("path_to_data_BC")
        #self.save_hyperparameters("dataset_version")
        
        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None
        self.dataset_IC_train = None
        self.dataset_IC_val = None
        self.dataset_IC_test = None
        self.dataset_BC_train = None
        self.dataset_BC_val = None
        self.dataset_BC_test = None

        self.column_names = None
        self.target_names = None
        self.in_features = None
        self.out_features = None
        self.scaler_x = None
        self.scaler_y = None

    @staticmethod
    def load_data(path_to_data, sample_size) -> pd.DataFrame:
        return pd.read_parquet(path_to_data).sample(frac=sample_size, replace=False)

    def prepare_data(self) -> None:
        # https://github.com/Lightning-AI/lightning/issues/11528
        pass

    def setup(self) -> None:
        # * Part 1: Dataset of collocation points
        df = get_numeric(self.load_data(self.hparams.path_to_data, self.hparams.sample_size))
        
        x_train, x_val, x_test, y_train, y_val, y_test = train_val_test_split(
            df=df,
            ratio=(self.TRAIN_RATIO, self.VAL_RATIO, self.TEST_RATIO),
            target_columns=self.hparams.targets,
        )
        self.column_names = list(x_train.columns)
        self.target_names = list(y_train.columns)
        self.in_features = x_train.shape[1]
        self.out_features = y_train.shape[1]

        x_train_norm, x_val_norm, x_test_norm, self.scaler_x = normalize(
            x_train,
            x_val,
            x_test,
            norm_type="z-score",
        )

        # Multi-output requires response scaling
        y_train_norm, y_val_norm, y_test_norm, self.scaler_y = normalize(
            y_train,
            y_val,
            y_test,
            norm_type="z-score",
        )

        self.dataset_train = torch.utils.data.TensorDataset(
            torch.Tensor(x_train_norm),
            torch.Tensor(y_train_norm),
        )
        self.dataset_val = torch.utils.data.TensorDataset(
            torch.Tensor(x_val_norm),
            torch.Tensor(y_val_norm),
        )
        self.dataset_test = torch.utils.data.TensorDataset(
            torch.Tensor(x_test_norm),
            torch.Tensor(y_test_norm),
        )

        # * IC
        df_IC = get_numeric(
            self.load_data(
                self.hparams.path_to_data_IC,
                sample_size=self.hparams.sample_size,
            )
        )
        x_train_IC, x_val_IC, x_test_IC, y_train_IC, y_val_IC, y_test_IC = train_val_test_split(
            df=df_IC,
            ratio=(self.TRAIN_RATIO, self.VAL_RATIO, self.TEST_RATIO),
            target_columns=self.hparams.targets, # must be same -> change in dataset
        )
        self.dataset_IC_train = torch.utils.data.TensorDataset(
            torch.Tensor(self.scaler_x.transform(x_train_IC)),
            torch.Tensor(self.scaler_y.transform(y_train_IC)),
        )
        self.dataset_IC_val = torch.utils.data.TensorDataset(
            torch.Tensor(self.scaler_x.transform(x_val_IC)),
            torch.Tensor(self.scaler_y.transform(y_val_IC)),
        )
        self.dataset_IC_test = torch.utils.data.TensorDataset(
            torch.Tensor(self.scaler_x.transform(x_test_IC)),
            torch.Tensor(self.scaler_y.transform(y_test_IC)),
        )

        # * BC
        df_BC = get_numeric(
            self.load_data(
                self.hparams.path_to_data_BC,
                sample_size=self.hparams.sample_size,
            )
        )
        x_train_BC, x_val_BC, x_test_BC, y_train_BC, y_val_BC, y_test_BC = train_val_test_split(
            df=df_BC,
            ratio=(self.TRAIN_RATIO, self.VAL_RATIO, self.TEST_RATIO),
            target_columns=self.hparams.targets,
        )
        self.dataset_BC_train = torch.utils.data.TensorDataset(
            torch.Tensor(self.scaler_x.transform(x_train_BC)),
            torch.Tensor(self.scaler_y.transform(y_train_BC)),
        )
        self.dataset_BC_val = torch.utils.data.TensorDataset(
            torch.Tensor(self.scaler_x.transform(x_val_BC)),
            torch.Tensor(self.scaler_y.transform(y_val_BC)),
        )
        self.dataset_BC_test = torch.utils.data.TensorDataset(
            torch.Tensor(self.scaler_x.transform(x_test_BC)),
            torch.Tensor(self.scaler_y.transform(y_test_BC)),
        )
       
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        # Alternative: https://github.com/Lightning-AI/lightning/pull/4606
        return torch.utils.data.DataLoader(
            ConcatDatasets(
                [
                    self.dataset_train,
                    self.dataset_IC_train,
                    self.dataset_BC_train,
                ],
            ),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            ConcatDatasets(
                [
                    self.dataset_val,
                    self.dataset_IC_val,
                    self.dataset_BC_val,
                ],
            ),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            ConcatDatasets(
                [
                    self.dataset_test,
                    self.dataset_IC_test,
                    self.dataset_BC_test,
                ],
            ),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
        )
    
    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            ConcatDatasets(
                [
                    self.dataset_test,
                    self.dataset_IC_test,
                    self.dataset_BC_test,
                ],
            ),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
        )
