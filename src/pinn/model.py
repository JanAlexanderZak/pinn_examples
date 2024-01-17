from typing import Union, List, Dict

import pytorch_lightning as pl
import torch
import torchmetrics
import numpy as np


class PINNLosses:
    """ PINNLosses organizes all loss functions w.r.t data, IC and BC.
        AD derivatives should be calculated within model bcs. of performance.
        (otherwise the class is instantiates at every single step)
    """
    @staticmethod
    def loss_function_data(y_pred, y_train) -> float:
        return torch.mean((y_pred - y_train) ** 2)
    
    @staticmethod
    def loss_function_IC_BC(y_pred, y_train) -> float:
        return torch.mean((y_pred - y_train) ** 2)

    def loss_function_PDE(
        self,
        u_x,
    ) -> float:
        pass


class PINNRegressor(pl.LightningModule):
    """ .
    Ressources:
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
    """
    def __init__(
        self,
        hyper_parameters: Dict[str, Union[str, int, float, List]],
        in_features: int,
        out_features: int,
        column_names: List[str],
        target_names: List[str],
        #pde_version: str,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(hyper_parameters)
        self.save_hyperparameters("in_features")
        self.save_hyperparameters("out_features")
        self.save_hyperparameters("column_names")
        self.save_hyperparameters("target_names")
        #self.save_hyperparameters("pde_version")

        self.pinn_losses = PINNLosses()
        
        # Configure
        self.linears = self.configure_linears()
        self._log_hyperparams = True

        self.train_mse = torchmetrics.MeanSquaredError()
        self.eval_mse = torchmetrics.MeanSquaredError()
        self.train_mae = torchmetrics.MeanAbsoluteError()
        self.eval_mae = torchmetrics.MeanAbsoluteError()
        self.train_mape = torchmetrics.MeanAbsolutePercentageError()
        self.eval_mape = torchmetrics.MeanAbsolutePercentageError()
        self.train_r2 = torchmetrics.MultioutputWrapper(torchmetrics.R2Score(), self.hparams.out_features)
        self.eval_r2 = torchmetrics.MultioutputWrapper(torchmetrics.R2Score(), self.hparams.out_features)
    
    def configure_linears(self) -> torch.nn.modules.container.ModuleList:
        """ Automatically generates the 'list-of-layers' from given hyperparameters.

        Args:
            num_hidden_layers (int): .
            size_hidden_layers (int): .
            in_features (int): .
            out_features (int): .

        Returns:
            (ModuleList): List of linear layers.
        """
        # plus 1 bcs. num is really the in-out feature connection
        hidden_layers_list = np.repeat(self.hparams.size_hidden_layers, self.hparams.num_hidden_layers + 1)
        layers_list = np.array([self.hparams.in_features, *hidden_layers_list, self.hparams.out_features])
        
        linears = torch.nn.ModuleList([
            torch.nn.Linear(
                layers_list[i], layers_list[i + 1]
            ) for i in range(len(layers_list) - 1)
        ])

        for i in range(len(layers_list) -  1):
            self.hparams.layer_initialization(linears[i].weight.data, gain=1.0)
            torch.nn.init.zeros_(linears[i].bias.data)
        
        return linears
    
    def _shared_eval_step(self, eval_batch, eval_batch_idx):
        """ Pytorch lightning recommends a shared evaluation step.
            This step is executed on val/test steps.
            https://lightning.ai/docs/pytorch/stable/common/lightning_module.html

        Args:
            eval_batch (_type_): .
            eval_batch_idx (_type_): .

        Returns:
            (Tuple(float)): All loss types.
        """
        x_eval, y_eval = eval_batch[0]
        y_pred = self(x_eval)

        total_data = torch.Tensor([len(y_eval)])

        loss = self.pinn_losses.loss_function_data(y_pred, y_eval)
        return loss, self.eval_mse(y_pred, y_eval), self.eval_mae(y_pred, y_eval), self.eval_mape(y_pred, y_eval), self.eval_r2(y_pred, y_eval), total_data

    def configure_optimizers(self) -> Dict:
        """ Configures the optimizer automatically.
            Function name is prescribed by PyTorch Lightning.

        Returns:
            Dict: Dictionary of optimizer/sheduler setup.
        """
        optimizer = self.hparams.optimizer(
            list(self.parameters()),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = self.hparams.scheduler(
            optimizer=optimizer,
            mode="min",
            patience=self.hparams.scheduler_patience,
            verbose=True,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": self.hparams.scheduler_monitor,
        }
    
    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        """ May be enabled during eval step.
            https://pytorch-lightning.readthedocs.io/en/1.3.8/benchmarking/performance.html

        Args:
            epoch (int): .
        """
        optimizer.zero_grad(set_to_none=True)

    def forward(self, x):
        # https://forums.fast.ai/t/where-should-i-place-the-batch-normalization-layer-s/56825/2
        # https://stackoverflow.com/questions/59003591/how-to-implement-dropout-in-pytorch-and-where-to-apply-it
        
        for layer in range(len(self.linears) - 1):
            # This loop doesnt affect performance.
            x = self.hparams.activation_function()(self.linears[layer](x))
            if self.hparams.batch_normalization:
                if self.hparams.cuda:
                    x = torch.nn.BatchNorm1d(self.hparams.size_hidden_layers).cuda()(x)
            if self.hparams.dropout:
                x = torch.nn.Dropout(p=self.hparams.dropout_p, inplace=False)(x)

        output = self.linears[-1](x) # regression

        return output
    
    def training_step(self, train_batch, batch_idx) -> float:
        """ Training step consists of 5 parts:
            (1) Load data, IC, BC from train_batch
            (2) Data loss
            (3) IC/BC loss
            (4) PDE loss (w/ auto-diff)
            (5) Logging

        Args:
            train_batch (Dataloader): Given batch by dataloader.
            batch_idx: .

        Returns:
            float: overall loss.
        """
        # * Part 1: Load
        x_train, y_train = train_batch[0]
        x_train_IC, y_train_IC = train_batch[1]
        x_train_BC, y_train_BC = train_batch[2]

        # Enable gradient beforehand such that autograd can build DAG
        # https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
        x_train.requires_grad_(True)
        x_train_IC.requires_grad_(True)
        x_train_BC.requires_grad_(True)

        total_data = torch.Tensor([len(y_train)])
        total_IC = torch.Tensor([len(x_train_IC)])
        total_BC = torch.Tensor([len(x_train_BC)])
        
        # * Part 2: Data
        y_pred = self.forward(x_train)
        loss_data = self.pinn_losses.loss_function_data(y_pred, y_train) * self.hparams.loss_data_param

        # * Part 3: IC/BC
        y_pred_IC_X = self.forward(x_train_IC)
        y_pred_BC_X = self.forward(x_train_BC)

        loss_IC = self.pinn_losses.loss_function_IC_BC(y_pred_IC_X, y_train_IC) * self.hparams.loss_IC_param
        loss_BC = self.pinn_losses.loss_function_IC_BC(y_pred_BC_X, y_train_BC) * self.hparams.loss_BC_param

        # * Part 4: PDE
        u_x = torch.autograd.grad(
            outputs=y_pred,
            inputs=x_train,
            grad_outputs=torch.ones_like(y_pred),
            retain_graph=True,
            create_graph=True,
        )[0][:, 0].view(-1, 1)

        loss_PDE = self.pinn_losses.loss_function_PDE(
            u_x,
        )  * self.hparams.loss_PDE_param
        
        # * Part 5: Logging
        loss = loss_data + loss_IC + loss_BC + loss_PDE
        r2 = torch.Tensor(self.train_r2(y_pred, y_train))
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=self.hparams.batch_size)
        self.log("train_loss_data", loss_data, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=self.hparams.batch_size)
        self.log("train_loss_IC", loss_IC, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=self.hparams.batch_size)
        self.log("train_loss_BC", loss_BC, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=self.hparams.batch_size)
        self.log("train_loss_PDE", loss_PDE, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=self.hparams.batch_size)
        self.log("train_total_data", total_data, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,)
        self.log("train_total_data_IC", total_IC, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,)
        self.log("train_total_data_BC", total_BC, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,)
        self.log("train_mse", self.train_mse(y_pred, y_train), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,)
        self.log("train_mae", self.train_mae(y_pred, y_train), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,)
        self.log("train_mape", self.train_mape(y_pred, y_train), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,)
        return loss

    def validation_step(self, val_batch, val_batch_idx, dataloader_idx=0) -> float:
        """ Validation step.
            Doesnt compute full loss, since it is disabled due to performace.

        Args:
            val_batch (DataLoader): validation batch from dataloader.
            val_batch_idx: .
            dataloader_idx: . Defaults to 0.

        Returns:
            loss: quasi-loss w/o PDE.

        Ressources:
            https://github.com/Lightning-AI/lightning/issues/4487
            https://github.com/Lightning-AI/lightning/issues/13948
            https://github.com/Lightning-AI/lightning/issues/10287
        """
        # 
        loss, mse, mae, mape, r2, total_data = self._shared_eval_step(val_batch, val_batch_idx)

        # * Part 2: Logging
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=self.hparams.batch_size)
        self.log("val_total_data", total_data, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,)
        self.log("val_mse", mse, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,)
        self.log("val_mae", mae, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,)
        self.log("val_mape", mape, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,)
        return loss
    
    def test_step(self, val_batch, val_batch_idx, dataloader_idx=0) -> float:
        """ Fundamentally, the same issues occur as with validation_step.

        Args:
            val_batch (_type_): validation batch.
            val_batch_idx (_type_): validation batch id.
            dataloader_idx (int, optional): -. Defaults to 0.

        Returns:
            float: quasi-loss.
        """
        loss, mse, mae, mape, r2, total_data = self._shared_eval_step(val_batch, val_batch_idx)

        # * Part 2: Logging
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=self.hparams.batch_size)
        self.log("test_total_data", total_data, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,)
        self.log("test_mse", mse, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,)
        self.log("test_mae", mae, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,)
        self.log("test_mape", mape, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,)
        return loss

    def predict_step(self, pred_batch, batch_idx) -> float:
        x_pred  = pred_batch[0]
        y_pred = self.forward(x_pred)
        return y_pred
