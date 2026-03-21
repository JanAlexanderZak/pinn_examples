""" Executable example of 1D wave equation in PyTorch Lightning.
"""
import pytorch_lightning as pl
import torch
import numpy as np

from src.dl import DeepLearningArguments
from src.wave_eq_1d.model import WaveEq1DPINNRegressor
from src.wave_eq_1d.data_module import WaveEq1DPINNDataModule
from src.wave_eq_1d.generate_dataset import generate_dataset


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

    hyper_parameters = {
        "activation_function": torch.nn.Tanh,
        "layer_initialization": torch.nn.init.xavier_uniform_,
        "optimizer": torch.optim.Adam,
        "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "scheduler_patience": 100,
        "weight_decay": 1e-3,
        "scheduler_monitor": "train_loss",
        "learning_rate": 1e-4,
        "loss_BC_param": 1,
        "loss_PDE_param": 1,
        "num_hidden_layers": 8,
        "size_hidden_layers": 20,
        "dropout": False,
        "dropout_p": 0.1,
        "batch_normalization": False,
        "c": 1.0,  # wave speed
    }

    data_module = WaveEq1DPINNDataModule(
        path_to_data="./src/wave_eq_1d/data/",
        args=args,
    )
    data_module.setup()

    train_loader = data_module.train_dataloader()
    test_loader = data_module.test_dataloader()

    for idx, (x, BC) in enumerate(train_loader):
        print(x[0].shape)
        print(x[0])
        break

    model_summary = pl.callbacks.ModelSummary(max_depth=1)

    model = WaveEq1DPINNRegressor(
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
    u_pred = trainer.predict(model, dataloaders=test_loader,)
    print(len(u_pred))
    pred_path = (
        "./src/wave_eq_1d"
        f"/data/predictions/predictions_{epochs}.pkl"
    )
    torch.save(u_pred, pred_path)

if __name__ == "__main__":
    main(20000)
