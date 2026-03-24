""" Executable example of thick-walled cylinder (Lame problem) in PyTorch Lightning.
"""
import pytorch_lightning as pl
import torch
import numpy as np

from src.dl import DeepLearningArguments
from src.thick_walled_cylinder.model import CylinderPINNRegressor
from src.thick_walled_cylinder.data_module import (
    CylinderPINNDataModule,
)
from src.thick_walled_cylinder.generate_dataset import (
    generate_dataset,
)
from src.thick_walled_cylinder.visualization import visualize


def train_single(epochs, load_case="internal_pressure"):
    pl.seed_everything(6020)

    a = 1.0
    b = 2.0
    E = 1.0
    nu = 0.3
    p_i = 1.0
    p_o = 0.5

    path_to_data = f"./src/thick_walled_cylinder/data/{load_case}/"
    generate_dataset(
        load_case=load_case, path=path_to_data,
        a=a, b=b, E=E, nu=nu, p_i=p_i, p_o=p_o,
    )

    args = DeepLearningArguments(
        seed=6020,
        batch_size=100,
        max_epochs=epochs,
        min_epochs=100,
        num_workers=6,
        accelerator="cpu",
        devices=-1,
        sample_size=1,
        pin_memory=True,
        persistent_workers=True,
    )

    hyper_parameters = {
        "activation_function": torch.nn.Tanh,
        "layer_initialization": torch.nn.init.xavier_uniform_,
        "optimizer": torch.optim.Adam,
        "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "scheduler_patience": 500,
        "weight_decay": 1e-4,
        "scheduler_monitor": "train_loss",
        "learning_rate": 1e-3,
        "loss_BC_param": 20,
        "loss_PDE_param": 1,
        "num_hidden_layers": 5,
        "size_hidden_layers": 40,
        "dropout": False,
        "dropout_p": 0.1,
        "batch_normalization": False,
        "E": E,
        "nu_poisson": nu,
        "a": a,
        "b": b,
        "load_case": load_case,
    }

    data_module = CylinderPINNDataModule(
        path_to_data=path_to_data,
        args=args,
    )
    data_module.setup()

    train_loader = data_module.train_dataloader()
    test_loader = data_module.test_dataloader()

    for idx, (r, BC) in enumerate(train_loader):
        print(f"Collocation: {r[0].shape}")
        print(f"BC: {BC[0].shape}, {BC[1].shape}")
        break

    model = CylinderPINNRegressor(
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
    u_pred = trainer.predict(model, dataloaders=test_loader)
    print(len(u_pred))
    torch.save(
        u_pred,
        f"{path_to_data}/predictions_{epochs}.pkl",
    )
    visualize(
        model, load_case,
        a=a, b=b, E=E, nu=nu, p_i=p_i, p_o=p_o,
    )


LOAD_CASES = [
    "internal_pressure",
    "external_pressure",
    "combined_pressure",
]


def main(epochs):
    for load_case in LOAD_CASES:
        print(f"\n{'='*60}")
        print(f"Training: {load_case}")
        print(f"{'='*60}\n")
        train_single(epochs, load_case=load_case)


if __name__ == "__main__":
    main(12000)
