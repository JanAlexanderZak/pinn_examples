""" PyTorch Lightning model.
"""
from typing import List, Dict

import pytorch_lightning as pl
import torch
import torchmetrics
import numpy as np

from torcheval.metrics import R2Score


class RaissiPINNLosses:
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
        u_t: torch.Tensor,
        u_x: torch.Tensor,
        u_xx: torch.Tensor,
        nu: float,
    ) -> float:
        return torch.mean((u_t + y_pred * u_x - nu * u_xx) ** 2)


class RaissiPINNRegressor(pl.LightningModule):
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
        self.pinn_losses = RaissiPINNLosses()
        
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
        if self.hparams.optimizer == torch.optim.LBFGS:
            # LBFGS has no L2 regularization via. weight decay
            optimizer = self.hparams.optimizer(
                list(self.parameters()),
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
        pass

    def net(self, x, t):
        # This is actually a nececessity for autograd to build the graph
        return self.forward(torch.cat([x, t], dim=1))

    def training_step(self, train_batch, batch_idx) -> float:
        # * Part 1: Calculation
        
        x_train_BC, y_train_BC = train_batch[1]
        x_train_PDE = train_batch[0][0]

        # This is necessary bcs of autograd graph
        #x_u = torch.tensor(x_train_BC[:, 0:1], requires_grad=True).float()
        #t_u = torch.tensor(x_train_BC[:, 1:2], requires_grad=True).float()
        x_f = torch.tensor(x_train_PDE[:, 0:1], requires_grad=True).float()
        t_f = torch.tensor(x_train_PDE[:, 1:2], requires_grad=True).float()
        #u = torch.tensor(y_train_BC[:, 0:1], requires_grad=True).float()
       
        total_BC = int(torch.tensor(len(x_train_BC)))
        total_PDE = int(torch.tensor(len(x_train_PDE)))
        
        y_pred_BC = self.forward(x_train_BC)  # same as torch.cat
        loss_BC = self.pinn_losses.loss_function_IC_BC(y_pred=y_pred_BC, y_train=y_train_BC)

        # Or equivalent:
        #y_pred_BC = self.forward(x_u, t_u)  # same as torch.cat
        #loss_BC = self.pinn_losses.loss_function_IC_BC(y_pred=y_pred_BC, y_train=u)

        y_pred_PDE = self.net(x_f, t_f)
        u_t = torch.autograd.grad(
            outputs=y_pred_PDE,
            inputs=t_f,
            grad_outputs=torch.ones_like(y_pred_PDE),
            retain_graph=True,
            create_graph=True,
        )[0]

        u_x = torch.autograd.grad(
            outputs=y_pred_PDE,
            inputs=x_f,
            grad_outputs=torch.ones_like(y_pred_PDE),
            retain_graph=True,
            create_graph=True,
        )[0]

        u_xx = torch.autograd.grad(
            outputs=u_x,
            inputs=x_f,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True,
        )[0]

        loss_PDE = self.pinn_losses.loss_function_PDE(
            y_pred_PDE,
            u_t,
            u_x,
            u_xx,
            nu=self.hparams.nu,
        ) * self.hparams.loss_PDE_param

        loss = loss_PDE + loss_BC
        
        # * Part 2: Logging
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,)
        self.log("train_total_PDE", total_PDE, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,)
        self.log("train_total_BC", total_BC, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True,)
        self.log("train_loss_PDE", loss_PDE, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,)
        self.log("train_loss_BC", loss_BC, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,)
        
        if self.current_epoch % 100 == 0:
            print(
                'Iter %d, Loss: %.5e, Loss_BC: %.5e, Loss_PDE: %.5e' % (self.current_epoch, loss.item(), loss_BC.item(), loss_PDE.item())
            )
        return loss

    def validation_step(self, val_batch, val_batch_idx):
        pass
    
    def test_step(self, val_batch, val_batch_idx, dataloader_idx=0) -> float:
        pass

    def predict_step(self, pred_batch, batch_idx) -> float:
        u_pred = self.forward(pred_batch[0])
        return u_pred
