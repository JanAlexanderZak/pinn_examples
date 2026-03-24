""" Executable example of Kovasznay flow (steady 2D Navier-Stokes) in PyTorch Lightning.
"""
import os

import pytorch_lightning as pl
import torch
import numpy as np

from src.dl import DeepLearningArguments
from src.navier_stokes_kovasznay.model import KovasznayPINNRegressor
from src.navier_stokes_kovasznay.data_module import (
    KovasznayPINNDataModule,
)
from src.navier_stokes_kovasznay.generate_dataset import generate_dataset, SX, SY
from src.navier_stokes_kovasznay.visualization import main as visualize


def main(epochs):
    pl.seed_everything(6020)
    generate_dataset()
    args = DeepLearningArguments(
        seed=6020,
        batch_size=50,
        max_epochs=epochs,
        min_epochs=100,
        num_workers=6,
        accelerator="cpu",
        devices=-1,
        sample_size=1,
        pin_memory=True,
        persistent_workers=True,
    )

    Re = 20.0
    hyper_parameters = {
        "activation_function": torch.nn.Tanh,
        "layer_initialization": torch.nn.init.xavier_uniform_,
        "optimizer": torch.optim.Adam,
        "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "scheduler_patience": 1000,
        "weight_decay": 1e-3,
        "scheduler_monitor": "train_loss",
        "learning_rate": 1e-3,
        "loss_BC_param": 1,
        "loss_PDE_param": 1,
        "num_hidden_layers": 8,
        "size_hidden_layers": 20,
        "dropout": False,
        "dropout_p": 0.1,
        "batch_normalization": False,
        "nu": 1.0 / Re,
        "Sx": SX,
        "Sy": SY,
    }

    data_module = KovasznayPINNDataModule(
        path_to_data="./src/navier_stokes_kovasznay/data/",
        args=args,
    )
    data_module.setup()

    train_loader = data_module.train_dataloader()
    test_loader = data_module.test_dataloader()

    for idx, (x, BC) in enumerate(train_loader):
        print(f"Collocation: {x[0].shape}")
        print(f"BC: {BC[0].shape}, {BC[1].shape}")
        break

    model_summary = pl.callbacks.ModelSummary(max_depth=1)

    model = KovasznayPINNRegressor(
        hyper_parameters=hyper_parameters,
        in_features=data_module.in_features,
        out_features=data_module.out_features,
        column_names=data_module.column_names,
        target_names=data_module.target_names,
    )
    model.hparams.update(data_module.hparams)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        sync_batchnorm=args.sync_batchnorm,
        min_epochs=args.min_epochs,
    )
    print(dict(model.hparams))

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
    )
    uvp_pred = trainer.predict(model, dataloaders=test_loader)
    print(len(uvp_pred))
    predictions_dir = "./src/navier_stokes_kovasznay/data/predictions"
    os.makedirs(predictions_dir, exist_ok=True)
    torch.save(
        uvp_pred,
        f"{predictions_dir}/predictions_{epochs}.pkl",
    )
    visualize(epochs)

if __name__ == "__main__":
    main(15000)
