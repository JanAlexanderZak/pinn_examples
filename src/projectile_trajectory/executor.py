""" Executable example of projectile trajectory in PyTorch Lightning.
"""
import numpy as np
import pytorch_lightning as pl
import torch

from src.dl import DeepLearningArguments
from src.projectile_trajectory.model import ProjectilePINNRegressor
from src.projectile_trajectory.data_module import (
    ProjectilePINNDataModule,
)
from src.projectile_trajectory.generate_dataset import (
    generate_dataset, _flight_time,
)
from src.projectile_trajectory.visualization import visualize


def train_single(epochs, case="no_drag"):
    pl.seed_everything(6020)

    v0 = 10.0
    alpha = np.pi / 4
    g = 9.81
    beta = 0.0 if case == "no_drag" else 0.5

    path_to_data = f"./src/projectile_trajectory/data/{case}/"
    generate_dataset(
        case=case, path=path_to_data,
        v0=v0, alpha=alpha, g=g, beta=beta,
    )

    # Non-dimensional PDE coefficients
    T_ref = _flight_time(case, v0=v0, alpha=alpha, g=g, beta=beta)
    V_ref = v0
    L_ref = v0 * T_ref
    g_nd = g * T_ref / V_ref
    beta_nd = beta * T_ref

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
        "loss_IC_param": 10,
        "loss_PDE_param": 1,
        "num_hidden_layers": 4,
        "size_hidden_layers": 30,
        "dropout": False,
        "dropout_p": 0.1,
        "batch_normalization": False,
        "g": g_nd,
        "beta": beta_nd,
        "v0": v0,
        "alpha": alpha,
        "case": case,
        "T_ref": T_ref,
        "V_ref": V_ref,
        "L_ref": L_ref,
    }

    data_module = ProjectilePINNDataModule(
        path_to_data=path_to_data,
        args=args,
    )
    data_module.setup()

    train_loader = data_module.train_dataloader()
    test_loader = data_module.test_dataloader()

    for idx, (t, IC) in enumerate(train_loader):
        print(f"Collocation: {t[0].shape}")
        print(f"IC: {IC[0].shape}, {IC[1].shape}")
        break

    model = ProjectilePINNRegressor(
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
    xy_pred = trainer.predict(model, dataloaders=test_loader)
    print(len(xy_pred))
    torch.save(
        xy_pred,
        f"{path_to_data}/predictions_{epochs}.pkl",
    )
    visualize(
        model, case,
        v0=v0, alpha=alpha, g=g, beta=beta,
    )


CASES = [
    "no_drag",
    "linear_drag",
]


def main(epochs):
    for case in CASES:
        print(f"\n{'='*60}")
        print(f"Training: {case}")
        print(f"{'='*60}\n")
        train_single(epochs, case=case)


if __name__ == "__main__":
    main(8000)
