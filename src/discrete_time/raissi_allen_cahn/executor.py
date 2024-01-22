""" Executable example of Raissi's Allen-Cahn equation in PyTorch Lightning.
"""
import pytorch_lightning as pl
import torch

from src.dl import DeepLearningArguments
from src.discrete_time.raissi_allen_cahn.model import RaissiPINNRegressor
from src.discrete_time.raissi_allen_cahn.data_module import RaissiPINNDataModule


def main(epochs):
    args = DeepLearningArguments(
        seed=6020,
        batch_size=64,
        max_epochs=epochs,
        min_epochs=10000,
        num_workers=6,
        accelerator="cpu",
        devices=-1,
        sample_size=1,
        pin_memory=True,
        persistent_workers=True,
    )

    hyper_parameters = {
        "activation_function": torch.nn.Tanh,
        "layer_initialization": torch.nn.init.xavier_normal_, #torch.nn.init.xavier_normal_
        "optimizer": torch.optim.LBFGS,
        "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "scheduler_patience": 200,
        "weight_decay": 0,
        "scheduler_monitor": "train_loss",
        "learning_rate": 1e-4,  #1e-4 worked but slow
        "loss_BC_param": 1,
        "loss_IC_param": 100,  # 100 worked
        "num_hidden_layers": 4,
        "size_hidden_layers": 200,
        "dropout": False,
        "dropout_p": 0.1,
        "batch_normalization": False,
    }
    
    data_module = RaissiPINNDataModule(
        path_to_data="./src/discrete_time/raissi_allen_cahn/data/",
        args=args,
    )
    data_module.setup()
    
    train_loader = data_module.train_dataloader()
    #val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    for idx, (IC, BC) in enumerate(train_loader):
        print(IC[0].shape)
        print(IC[0])
        break

    # Callbacks
    logger = pl.loggers.TensorBoardLogger(
        save_dir="",
        name=f"21012024",
    )
    early_stopping = pl.callbacks.EarlyStopping("train_loss", patience=1000, verbose=True,)
    model_summary = pl.callbacks.ModelSummary(max_depth=1)

    model = RaissiPINNRegressor(
        hyper_parameters=hyper_parameters,
        in_features=data_module.in_features,
        out_features=data_module.out_features,
        column_names=data_module.column_names,
        target_names=data_module.target_names,
    )
    model.hparams.update(data_module.hparams)
    
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[early_stopping, model_summary],
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
    torch.save(u_pred, f"./src/discrete_time/raissi_allen_cahn/data/predictions/predictions_{hyper_parameters['learning_rate']}_{hyper_parameters['loss_IC_param']}_{hyper_parameters['num_hidden_layers']}_{hyper_parameters['size_hidden_layers']}.pkl")


if __name__ == "__main__":
    main(200000)
