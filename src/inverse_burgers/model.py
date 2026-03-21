""" PINN for inverse Burgers equation (parameter identification).

PDE: u_t + u*u_x = nu*u_xx
Inverse problem: nu (viscosity) is unknown and learned from sparse noisy data.

References:
    Raissi, M., Perdikaris, P., & Karniadakis, G.E. (2019).
        "Physics-informed neural networks: A deep learning framework for
        solving forward and inverse problems involving nonlinear partial
        differential equations." Journal of Computational Physics, 378, 686-707.
        https://doi.org/10.1016/j.jcp.2018.10.045
        Section 3.2: "Data-driven discovery of partial differential equations"

    Raissi, M., & Karniadakis, G.E. (2018).
        "Hidden physics models: Machine learning of nonlinear partial
        differential equations." Journal of Computational Physics, 357, 125-141.
        https://doi.org/10.1016/j.jcp.2017.11.039
"""
from typing import List, Dict

import pytorch_lightning as pl
import torch
import torchmetrics
import numpy as np

from torcheval.metrics import R2Score


class InverseBurgersPINNLosses:
    def __init__(self) -> None:
        pass

    @staticmethod
    def loss_function_data(y_pred, y_train) -> torch.Tensor:
        return torch.mean((y_pred - y_train) ** 2)

    @staticmethod
    def loss_function_IC_BC(y_pred, y_train) -> torch.Tensor:
        return torch.mean((y_pred - y_train) ** 2)

    @staticmethod
    def loss_function_obs(y_pred, y_obs) -> torch.Tensor:
        return torch.mean((y_pred - y_obs) ** 2)

    @staticmethod
    def loss_function_PDE(
        y_pred,
        u_t: torch.Tensor,
        u_x: torch.Tensor,
        u_xx: torch.Tensor,
        nu: torch.nn.parameter.Parameter,
    ) -> torch.Tensor:
        return torch.mean((u_t + y_pred * u_x - nu * u_xx) ** 2)


class InverseBurgersPINNRegressor(pl.LightningModule):
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
        self.pinn_losses = InverseBurgersPINNLosses()

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

        # Learnable PDE parameter (viscosity)
        self.nu = None
        self.nus = None

    def configure_linears(self) -> torch.nn.modules.container.ModuleList:
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
        # Initialize learnable viscosity parameter
        self.nu = torch.nn.Parameter(
            torch.tensor(
                [self.hparams.nu_initial], requires_grad=True,
            )
        )
        self.nus = []

        if self.hparams.optimizer == torch.optim.LBFGS:
            optimizer = self.hparams.optimizer(
                list(self.parameters()) + [self.nu],
                lr=self.hparams.learning_rate,
            )
        elif (
            self.hparams.optimizer == torch.optim.Adam
            or self.hparams.optimizer == torch.optim.RMSprop
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
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": self.hparams.scheduler_monitor,
        }

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
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
        pass

    def net(self, x, t):
        return self.forward(torch.cat([x, t], dim=1))

    def training_step(self, train_batch, batch_idx) -> torch.Tensor:
        # * Part 1: Calculation

        x_train_PDE = train_batch[0][0]
        x_train_IC_BC, y_train_IC_BC = train_batch[1]
        x_obs, u_obs = train_batch[2]

        # Separate inputs for autograd graph
        x_pde_x = x_train_PDE[:, 0:1].clone().detach().requires_grad_(True)
        x_pde_t = x_train_PDE[:, 1:2].clone().detach().requires_grad_(True)

        total_IC_BC = int(torch.tensor(len(x_train_IC_BC)))
        total_PDE = int(torch.tensor(len(x_train_PDE)))
        total_obs = int(torch.tensor(len(x_obs)))

        # IC/BC loss
        u_pred_IC_BC = self.forward(x_train_IC_BC)
        loss_IC_BC = self.pinn_losses.loss_function_IC_BC(
            y_pred=u_pred_IC_BC, y_train=y_train_IC_BC,
        )

        # Observation loss (sparse noisy data)
        u_pred_obs = self.forward(x_obs)
        loss_obs = self.pinn_losses.loss_function_obs(
            y_pred=u_pred_obs, y_obs=u_obs,
        )

        # PDE loss with learnable nu
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
            u_x,
            u_xx,
            nu=self.nu,
        ) * self.hparams.loss_PDE_param

        loss = (
            loss_PDE
            + loss_IC_BC * self.hparams.loss_IC_BC_param
            + loss_obs * self.hparams.loss_obs_param
        )

        # Track nu evolution
        self.nus.append(self.nu.item())

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
            "train_total_obs", total_obs,
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
        self.log(
            "train_loss_obs", loss_obs,
            on_step=False, on_epoch=True,
            prog_bar=True, sync_dist=True,
        )
        self.log(
            "train_param_nu", self.nu.item(),
            sync_dist=True, on_epoch=True, prog_bar=True,
        )

        if self.current_epoch % 100 == 0:
            print(
                'Iter %d, Loss: %.5e, Loss_IC_BC: %.5e,'
                ' Loss_PDE: %.5e, Loss_obs: %.5e, nu: %.6f' % (
                    self.current_epoch, loss.item(),
                    loss_IC_BC.item(), loss_PDE.item(),
                    loss_obs.item(), self.nu.item(),
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
