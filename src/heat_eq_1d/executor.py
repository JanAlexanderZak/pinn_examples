""" Executable example of 1D heat equation in PyTorch Lightning.
"""
import os

import pytorch_lightning as pl
import torch
import numpy as np

from src.dl import DeepLearningArguments
from src.heat_eq_1d.model import HeatEq1DPINNRegressor
from src.heat_eq_1d.data_module import HeatEq1DPINNDataModule
from src.heat_eq_1d.generate_dataset import generate_dataset, T_REF, U_REF
from src.heat_eq_1d.visualization import main as visualize


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
        "alpha": 0.1 * T_REF,  # alpha_nd = alpha * T_ref / L_ref^2 = 0.5
        "source_coeff": 2.0 * T_REF / U_REF,  # source_nd = 2 * T_ref / U_ref = 5.0
        "T_ref": T_REF,
        "U_ref": U_REF,
    }

    data_module = HeatEq1DPINNDataModule(
        path_to_data="./src/heat_eq_1d/data/",
        args=args,
    )
    data_module.setup()

    train_loader = data_module.train_dataloader()
    #val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    for idx, (x, BC) in enumerate(train_loader):
        print(x[0].shape)
        print(x[0])
        break

    # Callbacks
    # logger = pl.loggers.TensorBoardLogger(
    #     save_dir="",
    #     name=(
    #         f"{ISO_DATE}_drop{hyper_parameters['dropout']}"
    #         f"_bn{hyper_parameters['batch_normalization']}"
    #         f"_dataloss{hyper_parameters['loss_data_param']}"
    #     ),
    # )
    # early_stopping = pl.callbacks.EarlyStopping(
    #     "train_param_mu", patience=3000, verbose=True,
    # )
    model_summary = pl.callbacks.ModelSummary(max_depth=1)

    model = HeatEq1DPINNRegressor(
        hyper_parameters=hyper_parameters,
        in_features=data_module.in_features,
        out_features=data_module.out_features,
        column_names=data_module.column_names,
        target_names=data_module.target_names,
    )
    model.hparams.update(data_module.hparams)

    trainer = pl.Trainer(
        #callbacks=[checkpoint_every_n_steps],
        max_epochs=args.max_epochs,
        sync_batchnorm=args.sync_batchnorm,
        min_epochs=args.min_epochs,
        #default_root_dir="./src/models",
        #val_check_interval=1.0,
    )
    print(dict(model.hparams))

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        #val_dataloaders=val_loader,
    )
    #print(trainer.test(model=model, dataloaders=test_loader,))
    u_pred = trainer.predict(model, dataloaders=test_loader,)
    print(len(u_pred))
    predictions_dir = "./src/heat_eq_1d/data/predictions"
    os.makedirs(predictions_dir, exist_ok=True)
    torch.save(
        u_pred,
        f"{predictions_dir}/predictions_{epochs}.pkl",
    )
    visualize(epochs)

if __name__ == "__main__":
    main(10000)
