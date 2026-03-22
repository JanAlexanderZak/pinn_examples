""" Executable example of inverse Burgers equation in PyTorch Lightning.

Learns the viscosity parameter nu from sparse noisy observations.
True value: nu = 0.01/pi ≈ 0.003183
"""
import os

import pytorch_lightning as pl
import torch
import numpy as np

from src.dl import DeepLearningArguments
from src.inverse_burgers.model import InverseBurgersPINNRegressor
from src.inverse_burgers.data_module import InverseBurgersPINNDataModule
from src.inverse_burgers.generate_dataset import generate_dataset
from src.inverse_burgers.visualization import main as visualize


def main(epochs):
    pl.seed_everything(6020)
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

    hyper_parameters = {
        "activation_function": torch.nn.Tanh,
        "layer_initialization": torch.nn.init.xavier_uniform_,
        "optimizer": torch.optim.Adam,
        "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "scheduler_patience": 1000,
        "weight_decay": 1e-3,
        "scheduler_monitor": "train_loss",
        "learning_rate": 1e-3,
        "loss_IC_BC_param": 1,
        "loss_PDE_param": 1,
        "loss_obs_param": 1,
        "num_hidden_layers": 8,
        "size_hidden_layers": 20,
        "dropout": False,
        "dropout_p": 0.1,
        "batch_normalization": False,
        "nu_initial": 0.1,  # Initial guess (far from true value 0.01/pi)
        "nu_true": 0.01 / np.pi,  # For reference only
    }

    path_to_data = "./src/inverse_burgers/data/"
    if not os.path.exists(os.path.join(path_to_data, "x_train_IC_BC.npy")):
        generate_dataset(path_to_data)

    data_module = InverseBurgersPINNDataModule(
        path_to_data=path_to_data,
        args=args,
    )
    data_module.setup()

    train_loader = data_module.train_dataloader()
    test_loader = data_module.test_dataloader()

    for idx, (x, IC_BC, obs) in enumerate(train_loader):
        print(f"Collocation: {x[0].shape}")
        print(f"IC/BC: {IC_BC[0].shape}, {IC_BC[1].shape}")
        print(f"Observations: {obs[0].shape}, {obs[1].shape}")
        break

    model_summary = pl.callbacks.ModelSummary(max_depth=1)

    model = InverseBurgersPINNRegressor(
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

    print(f"\nTrue nu:      {0.01/np.pi:.6f}")
    print(f"Learned nu:   {model.nu.item():.6f}")
    rel_err = abs(model.nu.item() - 0.01/np.pi) / (0.01/np.pi) * 100
    print(f"Relative err: {rel_err:.2f}%")

    u_pred = trainer.predict(model, dataloaders=test_loader)
    print(len(u_pred))

    pred_dir = "./src/inverse_burgers/data/predictions"
    torch.save(
        u_pred, f"{pred_dir}/predictions_{epochs}.pkl",
    )
    torch.save(
        model.nus, f"{pred_dir}/nus_{epochs}.pkl",
    )
    visualize(epochs)


if __name__ == "__main__":
    main(20500)
