""" PINN for projectile trajectory with optional linear drag.

Governing ODEs:
    No drag:     x_tt = 0,            y_tt = -g
    Linear drag: x_tt = -beta * x_t,  y_tt = -g - beta * y_t

Multi-output network: input t -> outputs (x, y).
Velocities and accelerations are derived via automatic differentiation.

Initial conditions are enforced via loss terms matching position (order 0)
and velocity (order 1) at t=0 for each output component (x=index 0, y=index 1).

References:
    Meriam, J.L. & Kraige, L.G. (2012).
        "Engineering Mechanics: Dynamics." Wiley, 7th edition.

    Raissi, M., Perdikaris, P., & Karniadakis, G.E. (2019).
        "Physics-informed neural networks: A deep learning framework for
        solving forward and inverse problems involving nonlinear partial
        differential equations." Journal of Computational Physics, 378, 686-707.
        https://doi.org/10.1016/j.jcp.2018.10.045
"""
from typing import List, Dict

import pytorch_lightning as pl
import torch
import torchmetrics
import numpy as np

from torcheval.metrics import R2Score


class ProjectilePINNLosses:
    def __init__(self) -> None:
        pass

    @staticmethod
    def loss_function_PDE(
        x_tt: torch.Tensor,
        y_tt: torch.Tensor,
        x_t: torch.Tensor,
        y_t: torch.Tensor,
        g: float,
        beta: float,
    ) -> torch.Tensor:
        """PDE residual for projectile equations of motion.

        No drag (beta=0):     f_x = x_tt,  f_y = y_tt + g
        Linear drag (beta>0): f_x = x_tt + beta*x_t,  f_y = y_tt + g + beta*y_t
        """
        f_x = x_tt + beta * x_t
        f_y = y_tt + g + beta * y_t
        return torch.mean(f_x ** 2) + torch.mean(f_y ** 2)

    @staticmethod
    def loss_function_IC(
        derivs: Dict,
        ic_orders: torch.Tensor,
        ic_output_indices: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        """IC loss matching predicted values/velocities to prescribed values.

        Args:
            derivs: {(order, output_idx): prediction_tensor} at IC points.
            ic_orders: Integer tensor of derivative orders (0=position, 1=velocity).
            ic_output_indices: Integer tensor of output indices (0=x, 1=y).
            y_true: Target values, shape (n_ic, 1).
        """
        loss = torch.tensor(0.0, device=y_true.device)
        for order in [0, 1]:
            for out_idx in [0, 1]:
                mask = (ic_orders == order) & (ic_output_indices == out_idx)
                if mask.any():
                    pred = derivs[(order, out_idx)][mask]
                    target = y_true[mask]
                    loss = loss + torch.mean((pred - target) ** 2)
        return loss


class ProjectilePINNRegressor(pl.LightningModule):
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

        self.pinn_losses = ProjectilePINNLosses()

        self.linears = self.configure_linears()
        self.activation = self.hparams.activation_function()
        self._log_hyperparams = True

        if self.hparams.batch_normalization:
            self.batch_norms = torch.nn.ModuleList([
                torch.nn.BatchNorm1d(self.hparams.size_hidden_layers)
                for _ in range(self.hparams.num_hidden_layers + 1)
            ])
        if self.hparams.dropout:
            self.dropout_layer = torch.nn.Dropout(p=self.hparams.dropout_p)

        self.train_mse = torchmetrics.MeanSquaredError()
        self.train_mae = torchmetrics.MeanAbsoluteError()

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

        for i in range(len(layers_list) - 1):
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

    def net(self, t):
        """Forward pass with time input for autograd."""
        return self.forward(t)

    def _compute_derivatives(self, t):
        """Compute position, velocity, acceleration from the network.

        Args:
            t: Input tensor with requires_grad=True, shape (n, 1).

        Returns:
            (xy, xy_t, xy_tt): Position (n,2), velocity (n,2), acceleration (n,2).
        """
        ones_col = torch.ones(t.shape[0], 1, device=t.device)
        xy = self.net(t)  # (n, 2): [x, y]

        # Velocity: d(xy)/dt
        x_t = torch.autograd.grad(
            outputs=xy[:, 0:1], inputs=t,
            grad_outputs=ones_col,
            retain_graph=True, create_graph=True,
        )[0]
        y_t = torch.autograd.grad(
            outputs=xy[:, 1:2], inputs=t,
            grad_outputs=ones_col,
            retain_graph=True, create_graph=True,
        )[0]

        # Acceleration: d2(xy)/dt2
        x_tt = torch.autograd.grad(
            outputs=x_t, inputs=t,
            grad_outputs=ones_col,
            retain_graph=True, create_graph=True,
        )[0]
        y_tt = torch.autograd.grad(
            outputs=y_t, inputs=t,
            grad_outputs=ones_col,
            retain_graph=True, create_graph=True,
        )[0]

        xy_t = torch.cat([x_t, y_t], dim=1)
        xy_tt = torch.cat([x_tt, y_tt], dim=1)

        return xy, xy_t, xy_tt

    def _shared_eval_step(self, eval_batch, eval_batch_idx):
        pass

    def training_step(self, train_batch, batch_idx) -> torch.Tensor:
        # * Part 1: Unpack batch
        t_train_PDE = train_batch[0][0]
        x_train_IC, y_train_IC = train_batch[1]

        total_IC = int(torch.tensor(len(x_train_IC)))
        total_PDE = int(torch.tensor(len(t_train_PDE)))

        # * Part 2: PDE loss (equations of motion)
        t_pde = t_train_PDE[:, 0:1].clone().detach().requires_grad_(True)
        xy, xy_t, xy_tt = self._compute_derivatives(t_pde)

        loss_PDE = self.pinn_losses.loss_function_PDE(
            x_tt=xy_tt[:, 0:1],
            y_tt=xy_tt[:, 1:2],
            x_t=xy_t[:, 0:1],
            y_t=xy_t[:, 1:2],
            g=self.hparams.g,
            beta=self.hparams.beta,
        ) * self.hparams.loss_PDE_param

        # * Part 3: IC loss (position and velocity at t=0)
        t_ic = x_train_IC[:, 0:1].clone().detach().requires_grad_(True)
        ic_orders = x_train_IC[:, 1].long()
        ic_output_indices = x_train_IC[:, 2].long()

        xy_ic, xy_t_ic, _ = self._compute_derivatives(t_ic)

        # Build derivs dict: (order, output_idx) -> prediction
        derivs = {
            (0, 0): xy_ic[:, 0:1],    # x position
            (0, 1): xy_ic[:, 1:2],    # y position
            (1, 0): xy_t_ic[:, 0:1],  # x velocity
            (1, 1): xy_t_ic[:, 1:2],  # y velocity
        }

        loss_IC = self.pinn_losses.loss_function_IC(
            derivs=derivs,
            ic_orders=ic_orders,
            ic_output_indices=ic_output_indices,
            y_true=y_train_IC,
        ) * self.hparams.loss_IC_param

        loss = loss_PDE + loss_IC

        # * Part 4: Logging
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
            "train_total_IC", total_IC,
            on_step=False, on_epoch=True,
            prog_bar=False, sync_dist=True,
        )
        self.log(
            "train_loss_PDE", loss_PDE,
            on_step=False, on_epoch=True,
            prog_bar=True, sync_dist=True,
        )
        self.log(
            "train_loss_IC", loss_IC,
            on_step=False, on_epoch=True,
            prog_bar=True, sync_dist=True,
        )

        if self.current_epoch % 100 == 0:
            print(
                'Iter %d, Loss: %.5e, Loss_IC: %.5e,'
                ' Loss_PDE: %.5e' % (
                    self.current_epoch, loss.item(),
                    loss_IC.item(), loss_PDE.item(),
                )
            )
        return loss

    def validation_step(self, val_batch, val_batch_idx):
        pass

    def test_step(self, val_batch, val_batch_idx, dataloader_idx=0) -> torch.Tensor:
        pass

    def predict_step(self, pred_batch, batch_idx) -> torch.Tensor:
        xy_pred = self.forward(pred_batch[0])
        return xy_pred

    def compute_pde_residual(self, t_candidates: torch.Tensor) -> torch.Tensor:
        """Compute absolute PDE residual for adaptive refinement (RAR callback).

        Args:
            t_candidates: Candidate points, shape (n, 1).

        Returns:
            Absolute residual, shape (n, 1).
        """
        t = t_candidates[:, 0:1].clone().detach().requires_grad_(True)
        xy, xy_t, xy_tt = self._compute_derivatives(t)

        f_x = xy_tt[:, 0:1] + self.hparams.beta * xy_t[:, 0:1]
        f_y = xy_tt[:, 1:2] + self.hparams.g + self.hparams.beta * xy_t[:, 1:2]
        residual = f_x.abs() + f_y.abs()
        return residual.detach()
