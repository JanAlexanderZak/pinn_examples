""" PINN for the 1D heat equation (continuous-time).

PDE: u_t = alpha * u_xx + sigma(x)  where sigma = 2*sin(pi*x)

References:
    Raissi, M., Perdikaris, P., & Karniadakis, G.E. (2019).
        "Physics-informed neural networks: A deep learning framework for
        solving forward and inverse problems involving nonlinear partial
        differential equations." Journal of Computational Physics, 378, 686-707.
        https://doi.org/10.1016/j.jcp.2018.10.045

    Free University of Brussels MOOC:
        "Solving PDEs - The Heat Equation (Explicit Method)."
        https://aquaulb.github.io/book_solving_pde_mooc/solving_pde_mooc/notebooks/
        04_PartialDifferentialEquations/04_03_Diffusion_Explicit.html
"""
from typing import List, Dict

import pytorch_lightning as pl
import torch
import torchmetrics
import numpy as np

from torcheval.metrics import R2Score


class HeatEq1DPINNLosses:
    def __init__(self) -> None:
        pass

    @staticmethod
    def loss_function_data(y_pred, y_train) -> torch.Tensor:
        return torch.mean((y_pred - y_train) ** 2)

    @staticmethod
    def loss_function_IC_BC(y_pred, y_train) -> torch.Tensor:
        return torch.mean((y_pred - y_train) ** 2)

    @staticmethod
    def loss_function_PDE(
        y_pred,
        u_t: torch.Tensor,
        u_xx: torch.Tensor,
        alpha: float,
        x_domain: torch.Tensor,
    ) -> torch.Tensor:
        # source: 2.0 * np.sin(np.pi * x)
        return torch.mean(
            (
                alpha * u_xx
                + 2.0 * torch.sin(torch.pi * x_domain)
                - u_t
            ) ** 2
        )


class HeatEq1DPINNRegressor(pl.LightningModule):
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
        self.pinn_losses = HeatEq1DPINNLosses()

        self.linears = self.configure_linears()
        self.activation = self.hparams.activation_function()
        self._log_hyperparams = True

        # Optional regularization layers
        if self.hparams.batch_normalization:
            self.batch_norms = torch.nn.ModuleList([
                torch.nn.BatchNorm1d(self.hparams.size_hidden_layers)
                for _ in range(self.hparams.num_hidden_layers + 1)
            ])
        if self.hparams.dropout:
            self.dropout_layer = torch.nn.Dropout(p=self.hparams.dropout_p)

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
        hidden_layers_list = np.repeat(
            self.hparams.size_hidden_layers,
            self.hparams.num_hidden_layers + 1,
        )
        layers_list = np.array([
            self.hparams.in_features,
            *hidden_layers_list,
            self.hparams.out_features,
        ])

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
        elif (
            self.hparams.optimizer == torch.optim.Adam
        ) or (
            self.hparams.optimizer == torch.optim.RMSprop
        ):
            optimizer = self.hparams.optimizer(
                list(self.parameters()),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            raise ValueError(f"Unknown Optimizer: '{self.hparams.optimizer}'.")

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
        for layer in range(len(self.linears) - 1):
            x = self.activation(self.linears[layer](x))
            if self.hparams.batch_normalization:
                x = self.batch_norms[layer](x)
            if self.hparams.dropout:
                x = self.dropout_layer(x)

        output = self.linears[-1](x)

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

    def training_step(self, train_batch, batch_idx) -> torch.Tensor:
        # * Part 1: Calculation

        x_train_PDE = train_batch[0][0]
        x_train_IC_BC, y_train_IC_BC = train_batch[1]

        # Separate inputs for autograd graph (need requires_grad for PDE derivatives)
        x_pde_x = x_train_PDE[:, 0:1].clone().detach().requires_grad_(True)
        x_pde_t = x_train_PDE[:, 1:2].clone().detach().requires_grad_(True)

        total_IC_BC = int(torch.tensor(len(x_train_IC_BC)))
        total_PDE = int(torch.tensor(len(x_train_PDE)))

        u_pred_IC_BC = self.forward(x_train_IC_BC)
        loss_IC_BC = self.pinn_losses.loss_function_IC_BC(
            y_pred=u_pred_IC_BC, y_train=y_train_IC_BC,
        )

        u_pred_PDE = self.net(x_pde_x, x_pde_t)
        u_t = torch.autograd.grad(
            outputs=u_pred_PDE,
            inputs=x_pde_t,
            grad_outputs=torch.ones_like(u_pred_PDE),
            retain_graph=True,
            create_graph=True,
        )[0]

        u_x = torch.autograd.grad(
            outputs=u_pred_PDE,
            inputs=x_pde_x,
            grad_outputs=torch.ones_like(u_pred_PDE),
            retain_graph=True,
            create_graph=True,
        )[0]

        u_xx = torch.autograd.grad(
            outputs=u_x,
            inputs=x_pde_x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True,
        )[0]

        loss_PDE = self.pinn_losses.loss_function_PDE(
            u_pred_PDE,
            u_t,
            u_xx,
            self.hparams.alpha,
            x_pde_x,
        ) * self.hparams.loss_PDE_param

        loss = loss_PDE + loss_IC_BC

        # * Part 2: Logging
        self.log(
            "train_loss", loss,
            on_step=False, on_epoch=True,
            prog_bar=True, sync_dist=True,
        )
        self.log(
            "train_total_PDE", total_PDE,
            on_step=False, on_epoch=True,
            prog_bar=False, sync_dist=True,
        )
        self.log(
            "train_total_IC_BC", total_IC_BC,
            on_step=False, on_epoch=True,
            prog_bar=False, sync_dist=True,
        )
        self.log(
            "train_loss_PDE", loss_PDE,
            on_step=False, on_epoch=True,
            prog_bar=True, sync_dist=True,
        )
        self.log(
            "train_loss_IC_BC", loss_IC_BC,
            on_step=False, on_epoch=True,
            prog_bar=True, sync_dist=True,
        )

        if self.current_epoch % 100 == 0:
            print(
                'Iter %d, Loss: %.5e, Loss_IC_BC: %.5e,'
                ' Loss_PDE: %.5e' % (
                    self.current_epoch, loss.item(),
                    loss_IC_BC.item(), loss_PDE.item(),
                )
            )
        return loss

    def validation_step(self, val_batch, val_batch_idx):
        pass

    def test_step(self, val_batch, val_batch_idx, dataloader_idx=0) -> torch.Tensor:
        pass

    def predict_step(self, pred_batch, batch_idx) -> torch.Tensor:
        u_pred = self.forward(pred_batch[0])
        return u_pred
