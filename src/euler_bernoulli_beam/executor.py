""" Executable example of Euler-Bernoulli beam deflection in PyTorch Lightning.
"""
import pytorch_lightning as pl
import torch
import numpy as np

from src.dl import DeepLearningArguments
from src.euler_bernoulli_beam.model import BeamPINNRegressor
from src.euler_bernoulli_beam.data_module import (
    BeamPINNDataModule,
)
from src.euler_bernoulli_beam.generate_dataset import (
    generate_dataset,
)
from src.euler_bernoulli_beam.visualization import visualize


def main(epochs, load_case="cantilever_point_load"):
    pl.seed_everything(6020)

    path_to_data = "./src/euler_bernoulli_beam/data/"
    generate_dataset(load_case=load_case, path=path_to_data)

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

    # q = 0 for point load (enters via shear BC), q0 for distributed load cases
    q = 0.0 if load_case == "cantilever_point_load" else 1.0

    hyper_parameters = {
        "activation_function": torch.nn.Tanh,
        "layer_initialization": torch.nn.init.xavier_uniform_,
        "optimizer": torch.optim.Adam,
        "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "scheduler_patience": 500,
        "weight_decay": 1e-4,
        "scheduler_monitor": "train_loss",
        "learning_rate": 1e-3,
        "loss_BC_param": 10,
        "loss_PDE_param": 1,
        "num_hidden_layers": 6,
        "size_hidden_layers": 40,
        "dropout": False,
        "dropout_p": 0.1,
        "batch_normalization": False,
        "EI": 1.0,
        "q": q,
        "load_case": load_case,
    }

    data_module = BeamPINNDataModule(
        path_to_data=path_to_data,
        args=args,
    )
    data_module.setup()

    train_loader = data_module.train_dataloader()
    test_loader = data_module.test_dataloader()

    for idx, (x, BC) in enumerate(train_loader):
        print(f"Collocation: {x[0].shape}")
        print(f"BC: {BC[0].shape}, {BC[1].shape}")
        break

    model = BeamPINNRegressor(
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
    w_pred = trainer.predict(model, dataloaders=test_loader)
    print(len(w_pred))
    torch.save(
        w_pred,
        f"./src/euler_bernoulli_beam/data/predictions_{epochs}.pkl",
    )
    visualize(model, load_case)


if __name__ == "__main__":
    main(15000)
