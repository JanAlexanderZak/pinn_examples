""" Executable example of Moseley's harmonic oscillator in PyTorch Lightning.
"""
import pytorch_lightning as pl
import torch

from src.dl import DeepLearningArguments
from src.continuous_time.moseley_oscillator.model import MoseleyPINNLosses, MoseleyPINNRegressor
from src.continuous_time.moseley_oscillator.data_module import MoseleyPINNDataModule


def main():
    args = DeepLearningArguments(
        seed=6020,
        batch_size=50,
        max_epochs=100000,
        min_epochs=500,
        num_workers=6,
        accelerator="cpu",
        devices=-1,
        sample_size=1,
        pin_memory=False,
    )

    hyper_parameters = {
        "activation_function": torch.nn.Tanh,
        "layer_initialization": torch.nn.init.xavier_uniform_, #torch.nn.init.xavier_normal_
        "optimizer": torch.optim.Adam,
        "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "scheduler_patience": 1000,
        "weight_decay": 1e-3,
        "scheduler_monitor": "train_param_mu",
        "learning_rate": 1e-3,
        "t_domain": torch.linspace(0, 1, 30).view(-1, 1).requires_grad_(True),
        "loss_IC_param": 1,
        "loss_BC_param": 1,
        "loss_PDE_param": 0.1,
        "num_hidden_layers": 6,
        "size_hidden_layers": 64,
        "dropout": False,
        "dropout_p": 0.1,
        "batch_normalization": False,
        "d": 2,
        "w0": 20,
    }

    data_module = MoseleyPINNDataModule(
        path_to_data="./src/continuous_time/moseley_oscillator/data/dataset.parquet",
        targets=["u_obs"],
        args=args,
    )
    data_module.setup()
    
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    #test_loader = data_module.test_dataloader()

    for i, (x_train, y_train) in enumerate(train_loader):
        print(x_train.shape, y_train.shape)
        print(x_train, y_train)
        break

    # Callbacks
    # logger = pl.loggers.TensorBoardLogger(
    #     save_dir="", 
    #     name=f"{ISO_DATE}_drop{hyper_parameters['dropout']}_bn{hyper_parameters['batch_normalization']}_dataloss{hyper_parameters['loss_data_param']}",
    # )
    # early_stopping = pl.callbacks.EarlyStopping("train_param_mu", patience=3000, verbose=True,)
    model_summary = pl.callbacks.ModelSummary(max_depth=1)

    model = MoseleyPINNRegressor(
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
        val_dataloaders=val_loader,
    )
    #print(trainer.test(model=model, dataloaders=test_loader,))
    #print(trainer.predict(dataloaders=test_loader,))


if __name__ == "__main__":
    main()
