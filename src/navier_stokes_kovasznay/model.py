""" PINN for Kovasznay flow (steady 2D incompressible Navier-Stokes).

PDE:
    u*u_x + v*u_y = -p_x + nu*(u_xx + u_yy)    (x-momentum)
    u*v_x + v*v_y = -p_y + nu*(v_xx + v_yy)    (y-momentum)
    u_x + v_y = 0                                (continuity)

Multi-output network: inputs (x, y) -> outputs (u, v, p)

References:
    Kovasznay, L.I.G. (1948).
        "Laminar flow behind a two-dimensional grid."
        Mathematical Proceedings of the Cambridge Philosophical Society,
        44(1), 58-62. https://doi.org/10.1017/S0305004100023999

    Raissi, M., Perdikaris, P., & Karniadakis, G.E. (2019).
        "Physics-informed neural networks: A deep learning framework for
        solving forward and inverse problems involving nonlinear partial
        differential equations." Journal of Computational Physics, 378, 686-707.
        https://doi.org/10.1016/j.jcp.2018.10.045

    Jin, X., Cai, S., Li, H., & Karniadakis, G.E. (2021).
        "NSFnets (Navier-Stokes flow nets): Physics-informed neural networks
        for the incompressible Navier-Stokes equations."
        Journal of Computational Physics, 426, 109951.
        https://doi.org/10.1016/j.jcp.2020.109951
"""
from typing import List, Dict

import pytorch_lightning as pl
import torch
import torchmetrics
import numpy as np

from torcheval.metrics import R2Score


class KovasznayPINNLosses:
    def __init__(self) -> None:
        pass

    @staticmethod
    def loss_function_BC(y_pred, y_train) -> torch.Tensor:
        return torch.mean((y_pred - y_train) ** 2)

    @staticmethod
    def loss_function_PDE(
        u, v,
        u_x, u_y, u_xx, u_yy,
        v_x, v_y, v_xx, v_yy,
        p_x, p_y,
        nu: float,
    ) -> torch.Tensor:
        # x-momentum: u*u_x + v*u_y + p_x - nu*(u_xx + u_yy) = 0
        f_u = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
        # y-momentum: u*v_x + v*v_y + p_y - nu*(v_xx + v_yy) = 0
        f_v = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
        # continuity: u_x + v_y = 0
        f_c = u_x + v_y

        return torch.mean(f_u ** 2) + torch.mean(f_v ** 2) + torch.mean(f_c ** 2)


class KovasznayPINNRegressor(pl.LightningModule):
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
        self.pinn_losses = KovasznayPINNLosses()

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
        if self.hparams.optimizer == torch.optim.LBFGS:
            optimizer = self.hparams.optimizer(
                list(self.parameters()),
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

    def net(self, x, y):
        """ Forward pass with separate x, y inputs for autograd.

        Returns:
            (u, v, p): velocity components and pressure
        """
        output = self.forward(torch.cat([x, y], dim=1))
        u = output[:, 0:1]
        v = output[:, 1:2]
        p = output[:, 2:3]
        return u, v, p

    def training_step(self, train_batch, batch_idx) -> torch.Tensor:
        # * Part 1: Calculation

        x_train_PDE = train_batch[0][0]
        x_train_BC, y_train_BC = train_batch[1]

        # Separate inputs for autograd graph
        x_pde_x = x_train_PDE[:, 0:1].clone().detach().requires_grad_(True)
        x_pde_y = x_train_PDE[:, 1:2].clone().detach().requires_grad_(True)

        total_BC = int(torch.tensor(len(x_train_BC)))
        total_PDE = int(torch.tensor(len(x_train_PDE)))

        # BC loss (multi-output)
        uvp_pred_BC = self.forward(x_train_BC)
        loss_BC = self.pinn_losses.loss_function_BC(
            y_pred=uvp_pred_BC, y_train=y_train_BC,
        )

        # PDE loss
        u, v, p = self.net(x_pde_x, x_pde_y)

        # Derivatives of u
        u_x = torch.autograd.grad(
            outputs=u, inputs=x_pde_x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True, create_graph=True,
        )[0]
        u_y = torch.autograd.grad(
            outputs=u, inputs=x_pde_y,
            grad_outputs=torch.ones_like(u),
            retain_graph=True, create_graph=True,
        )[0]
        u_xx = torch.autograd.grad(
            outputs=u_x, inputs=x_pde_x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True, create_graph=True,
        )[0]
        u_yy = torch.autograd.grad(
            outputs=u_y, inputs=x_pde_y,
            grad_outputs=torch.ones_like(u_y),
            retain_graph=True, create_graph=True,
        )[0]

        # Derivatives of v
        v_x = torch.autograd.grad(
            outputs=v, inputs=x_pde_x,
            grad_outputs=torch.ones_like(v),
            retain_graph=True, create_graph=True,
        )[0]
        v_y = torch.autograd.grad(
            outputs=v, inputs=x_pde_y,
            grad_outputs=torch.ones_like(v),
            retain_graph=True, create_graph=True,
        )[0]
        v_xx = torch.autograd.grad(
            outputs=v_x, inputs=x_pde_x,
            grad_outputs=torch.ones_like(v_x),
            retain_graph=True, create_graph=True,
        )[0]
        v_yy = torch.autograd.grad(
            outputs=v_y, inputs=x_pde_y,
            grad_outputs=torch.ones_like(v_y),
            retain_graph=True, create_graph=True,
        )[0]

        # Derivatives of p
        p_x = torch.autograd.grad(
            outputs=p, inputs=x_pde_x,
            grad_outputs=torch.ones_like(p),
            retain_graph=True, create_graph=True,
        )[0]
        p_y = torch.autograd.grad(
            outputs=p, inputs=x_pde_y,
            grad_outputs=torch.ones_like(p),
            retain_graph=True, create_graph=True,
        )[0]

        loss_PDE = self.pinn_losses.loss_function_PDE(
            u, v,
            u_x, u_y, u_xx, u_yy,
            v_x, v_y, v_xx, v_yy,
            p_x, p_y,
            nu=self.hparams.nu,
        ) * self.hparams.loss_PDE_param

        loss = loss_PDE + loss_BC

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
            "train_total_BC", total_BC,
            on_step=False, on_epoch=True,
            prog_bar=False, sync_dist=True,
        )
        self.log(
            "train_loss_PDE", loss_PDE,
            on_step=False, on_epoch=True,
            prog_bar=True, sync_dist=True,
        )
        self.log(
            "train_loss_BC", loss_BC,
            on_step=False, on_epoch=True,
            prog_bar=True, sync_dist=True,
        )

        if self.current_epoch % 100 == 0:
            print(
                'Iter %d, Loss: %.5e, Loss_BC: %.5e,'
                ' Loss_PDE: %.5e' % (
                    self.current_epoch, loss.item(),
                    loss_BC.item(), loss_PDE.item(),
                )
            )
        return loss

    def validation_step(self, val_batch, val_batch_idx):
        pass

    def test_step(self, val_batch, val_batch_idx, dataloader_idx=0) -> torch.Tensor:
        pass

    def predict_step(self, pred_batch, batch_idx) -> torch.Tensor:
        uvp_pred = self.forward(pred_batch[0])
        return uvp_pred
