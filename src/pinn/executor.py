import pytorch_lightning as pl
import torch

from src.dl import DeepLearningArguments
from src.pinn.model import PINNRegressor
from src.pinn.data_module import PINNDataModule
from src.callbacks import CheckpointEveryNSteps


def main():
    args = DeepLearningArguments(
        seed=1,
        batch_size=1,
        max_epochs=1,
        min_epochs=1,
        num_workers=1,
        accelerator="auto",
        devices=1,
        sample_size=1,
        pin_memory=False,
    )

    hyper_parameters = {
        "activation_function": torch.nn.Tanh,
        "layer_initialization": torch.nn.init.xavier_normal_,
        "optimizer": torch.optim.Adam,
        "weight_decay": 1,
        "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "scheduler_patience": 1,
        "scheduler_monitor": "train_loss",
        "learning_rate": 1,
        "loss_IC_param": 1,
        "loss_BC_param": 1,
        "loss_PDE_param": 1,
        "loss_data_param": 1,
        "num_hidden_layers": 1,
        "size_hidden_layers": 1,
        "dropout": True,
        "batch_normalization": False,
        "dropout_p": 0.1,
        "cuda": False,
    }

    # Complete setup with data args in Model
    data_module = PINNDataModule(
        path_to_data="./src/data/data.parquet",
        path_to_data_IC="./src/data/IC.parquet",
        path_to_data_BC_z="./src/data/BC.parquet",
        targets=["tgt"],
        args=args,
    )
    data_module.setup()
    
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    model = PINNRegressor(
        hyper_parameters=hyper_parameters,
        in_features=data_module.in_features,
        out_features=data_module.out_features,
        column_names=data_module.column_names,
        target_names=data_module.target_names,
    )
    model.hparams.update(data_module.hparams)

    checkpoint_every_n_steps = CheckpointEveryNSteps(save_step_frequency=1)
    
    trainer = pl.Trainer(
        callbacks=[checkpoint_every_n_steps],
        max_epochs=args.max_epochs,
        sync_batchnorm=args.sync_batchnorm,
        min_epochs=args.min_epochs,
        #default_root_dir="./src/pinn/models",
        val_check_interval=1.0,
    )
    print(dict(model.hparams))

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    #print(trainer.test(model=model, dataloaders=test_loader,))
    #print(trainer.predict(dataloaders=test_loader,))


if __name__ == "__main__":
    main()
