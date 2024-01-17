""" PyTorch Lightning model.
"""
from typing import Callable, Union, List, Dict

import pytorch_lightning as pl
import torch
import torchmetrics
import numpy as np
import matplotlib.pyplot as plt

from torcheval.metrics import R2Score


class MoseleyPINNLosses:
    def __init__(self) -> None:
        pass

    @staticmethod
    def loss_function_data(y_pred, y_train) -> float:
        return torch.mean((y_pred - y_train) ** 2)
    
    @staticmethod
    def loss_function_IC_BC(y_pred, y_train) -> float:
        return torch.mean((y_pred - y_train) ** 2)

    @staticmethod
    def loss_function_PDE(
        y_pred,
        u_x: torch.Tensor,
        u_xx: torch.Tensor,
        mu: torch.nn.parameter.Parameter,
        w0: int,
    ) -> float:
        return torch.mean((u_xx + mu * u_x + w0 * y_pred) ** 2)


class MoseleyPINNRegressor(pl.LightningModule):
    def __init__(
        self,
        hyper_parameters,
        in_features: int,
        out_features: int,
        column_names: List[str],
        target_names: List[str],
    ) -> None:
        super().__init__()

        self.save_hyperparameters(hyper_parameters)
        self.save_hyperparameters("in_features")
        self.save_hyperparameters("out_features")
        self.save_hyperparameters("column_names")
        self.save_hyperparameters("target_names")

        # Loss and Params
        self.pinn_losses = MoseleyPINNLosses()
        
        self.linears = self.configure_linears()
        self._log_hyperparams = True

        self.train_mse = torchmetrics.MeanSquaredError()
        self.eval_mse = torchmetrics.MeanSquaredError()
        self.train_mae = torchmetrics.MeanAbsoluteError()
        self.eval_mae = torchmetrics.MeanAbsoluteError()
        self.train_mape = torchmetrics.MeanAbsolutePercentageError()
        self.eval_mape = torchmetrics.MeanAbsolutePercentageError()
        self.train_r2 = R2Score()
        self.eval_r2 = R2Score()

        self.mu = None
        self.mus = None
        self.t_domain = self.hparams.t_domain
    
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
            self.hparams.layer_initialization(linears[i].weight.data)
            torch.nn.init.zeros_(linears[i].bias.data)
        
        return linears
    
    def configure_optimizers(self) -> Dict:
        """ Configures the optimizer automatically.
            Function name is prescribed by PyTorch Lightning.

        Returns:
            Dict: Dictionary of optimizer/sheduler setup.
        """
        self.mu = torch.nn.Parameter(torch.zeros(1, requires_grad=True))
        self.mus = []

        if self.hparams.optimizer == torch.optim.LBFGS:
            # LBFGS has no L2 regularization via. weight decay
            optimizer = self.hparams.optimizer(
                list(self.parameters()) + [self.mu],  #!
                lr=self.hparams.learning_rate,
            )
        elif (self.hparams.optimizer == torch.optim.Adam) or (self.hparams.optimizer == torch.optim.RMSprop): 
            optimizer = self.hparams.optimizer(
                list(self.parameters()),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            print(f"Unknown Optimizer: '{self.hparams.optimizer}'.")

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
            x = self.hparams.activation_function()(self.linears[layer](x))
            if self.hparams.batch_normalization:
                if self.hparams.cuda:
                    x = torch.nn.BatchNorm1d(self.hparams.size_hidden_layers).cuda()(x)
            if self.hparams.dropout:
                x = torch.nn.Dropout(p=self.hparams.dropout_p, inplace=False)(x)

        output = self.linears[-1](x) # regression

        return output

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
        x_eval, y_eval = eval_batch

        total_data = torch.Tensor([len(y_eval)])
        
        y_pred = self.forward(x_eval)
        loss_data = self.pinn_losses.loss_function_data(y_pred, y_eval)
        
        self.eval_r2.update(y_pred, y_eval)
        return loss_data, self.eval_mse(y_pred, y_eval), self.eval_mae(y_pred, y_eval), self.eval_mape(y_pred, y_eval), self.eval_r2.compute(), total_data

    def training_step(self, train_batch, batch_idx) -> float:
        # * Part 1: Calculation
        x_train, y_train = train_batch
        total = int(torch.tensor(len(y_train)))
        
        y_pred = self.forward(x_train)
        loss_data = self.pinn_losses.loss_function_data(y_pred=y_pred, y_train=y_train)

        y_pred_PDE = self.forward(self.hparams.t_domain)
        u_x = torch.autograd.grad(
            outputs=y_pred_PDE,
            inputs=self.hparams.t_domain,
            grad_outputs=torch.ones_like(y_pred_PDE),
            retain_graph=True,
            create_graph=True,
        )[0]

        u_xx = torch.autograd.grad(
            outputs=u_x,
            inputs=self.hparams.t_domain,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True,
        )[0]

        loss_PDE = self.pinn_losses.loss_function_PDE(
            y_pred_PDE,
            u_x,
            u_xx,
            mu=self.mu,  #!
            w0=self.hparams.w0,
        ) * self.hparams.loss_PDE_param

        loss = loss_PDE + loss_data
        
        # * Part 2: Logging
        self.train_r2.update(y_pred, y_train)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,)
        self.log("train_total", total, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,)
        self.log("train_mse", self.train_mse(y_pred, y_train), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,)
        self.log("train_r2", self.train_r2.compute(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,)
        self.log("train_mae", self.train_mae(y_pred, y_train), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,)
        self.log("train_mape", self.train_mape(y_pred, y_train), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,)  
        self.log("train_loss_PDE", loss_PDE, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,)
        self.log("train_loss_data", loss_data, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,)
        self.log("train_param_mu", self.mu.item(), sync_dist=True, on_epoch=True, prog_bar=True,)
        self.mus.append(self.mu.item())

        if self.current_epoch == 10000:
            t_test = torch.linspace(0, 1, 300).view(-1, 1)
            y_pred = self.forward(t_test).detach()
            plt.figure(figsize=(6, 2.5))
            plt.plot(t_test[:, 0], y_pred[:, 0], label="PINN solution", color="tab:green")
            plt.title(f"Training step {self.current_epoch}.")
            plt.legend()
            plt.show()

        return loss

    def validation_step(self, val_batch, val_batch_idx):
        # https://github.com/Lightning-AI/lightning/issues/4487
        # https://github.com/Lightning-AI/lightning/issues/13948
        # https://github.com/Lightning-AI/lightning/issues/10287
        # * Part 1: Calculation
        loss_data, mse, mae, mape, r2, total_data = self._shared_eval_step(val_batch, val_batch_idx)
        loss = loss_data

        # * Part 2: Logging
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=self.hparams.batch_size)
        self.log("val_loss_data", loss_data, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=self.hparams.batch_size)
        self.log("val_total_data", total_data, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,)
        self.log("val_mse", mse, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,)
        self.log("val_mae", mae, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,)
        self.log("val_mape", mape, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,)
        self.log("val_r2", r2, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,)
        return loss
    
    def test_step(self, val_batch, val_batch_idx, dataloader_idx=0) -> float:
        """ Test step.

        Args:
            val_batch (_type_): validation batch.
            val_batch_idx (_type_): validation batch id.
            dataloader_idx (int, optional): -. Defaults to 0.

        Returns:
            float: quasi-loss.
        """
        loss_data, mse, mae, mape, r2, total_data = self._shared_eval_step(val_batch, val_batch_idx)
        loss = loss_data

        # * Part 2: Logging
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=self.hparams.batch_size)
        self.log("test_loss_data", loss_data, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=self.hparams.batch_size)
        self.log("test_total_data", total_data, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,)
        self.log("test_mse", mse, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,)
        self.log("test_mae", mae, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,)
        self.log("test_mape", mape, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,)
        self.log("test_r2", r2, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,)
        return loss

    def predict_step(self, pred_batch, batch_idx) -> float:
        x_pred, _ = pred_batch
        return self.forward(x_pred)
