""" PINN for Euler-Bernoulli beam deflection (static, 1D).

Governing ODE (4th order):
    EI * d4w/dx4 = q(x)

Single-output network: input x -> output w(x).
Slope, moment, and shear are derived via automatic differentiation.

Boundary conditions are enforced via loss terms that match the predicted
derivative of the appropriate order to the prescribed value. The BC order
(0=w, 1=dw/dx, 2=d2w/dx2, 3=d3w/dx3) is encoded in the second column
of x_train_BC.

References:
    Timoshenko, S.P. & Gere, J.M. (1961).
        "Theory of Elastic Stability." McGraw-Hill.

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


class BeamPINNLosses:
    def __init__(self) -> None:
        pass

    @staticmethod
    def loss_function_PDE(
        w_xxxx: torch.Tensor,
        q: float,
        EI: float,
    ) -> torch.Tensor:
        """PDE residual: EI * d4w/dx4 - q(x) = 0."""
        residual = EI * w_xxxx - q
        return torch.mean(residual ** 2)

    @staticmethod
    def loss_function_BC(
        derivs: Dict[int, torch.Tensor],
        bc_orders: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        """BC loss matching predicted derivatives to prescribed values.

        Args:
            derivs: {0: w, 1: w_x, 2: w_xx, 3: w_xxx} at BC points.
            bc_orders: Integer tensor of derivative orders, shape (n_bc,).
            y_true: Target values, shape (n_bc, 1).
        """
        loss = torch.tensor(0.0, device=y_true.device)
        for order in [0, 1, 2, 3]:
            mask = (bc_orders == order)
            if mask.any():
                pred = derivs[order][mask]
                target = y_true[mask]
                loss = loss + torch.mean((pred - target) ** 2)
        return loss


class BeamPINNRegressor(pl.LightningModule):
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

        self.pinn_losses = BeamPINNLosses()

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

    def _shared_eval_step(self, eval_batch, eval_batch_idx):
        pass

    def net(self, x):
        """Forward pass with single input x for autograd."""
        return self.forward(x)

    def _compute_derivatives(self, x, up_to_order=4):
        """Compute derivatives of w w.r.t. x up to the given order.

        Args:
            x: Input tensor with requires_grad=True, shape (n, 1).
            up_to_order: Maximum derivative order (1-4).

        Returns:
            Tuple of (w, w_x, w_xx, w_xxx, w_xxxx) up to requested order.
        """
        ones = torch.ones_like(x)
        w = self.net(x)
        results = [w]

        prev = w
        for _ in range(up_to_order):
            d = torch.autograd.grad(
                outputs=prev, inputs=x,
                grad_outputs=ones,
                retain_graph=True, create_graph=True,
            )[0]
            results.append(d)
            prev = d

        return tuple(results)

    def training_step(self, train_batch, batch_idx) -> torch.Tensor:
        # * Part 1: Unpack batch
        x_train_PDE = train_batch[0][0]
        x_train_BC, y_train_BC = train_batch[1]

        total_BC = int(torch.tensor(len(x_train_BC)))
        total_PDE = int(torch.tensor(len(x_train_PDE)))

        # * Part 2: PDE loss (4th-order derivative)
        x_pde = x_train_PDE[:, 0:1].clone().detach().requires_grad_(True)
        w, w_x, w_xx, w_xxx, w_xxxx = self._compute_derivatives(x_pde, up_to_order=4)

        loss_PDE = self.pinn_losses.loss_function_PDE(
            w_xxxx=w_xxxx,
            q=self.hparams.q,
            EI=self.hparams.EI,
        ) * self.hparams.loss_PDE_param

        # * Part 3: BC loss (derivative BCs)
        x_bc_pos = x_train_BC[:, 0:1].clone().detach().requires_grad_(True)
        bc_orders = x_train_BC[:, 1:2].squeeze()

        w_bc, w_bc_x, w_bc_xx, w_bc_xxx = self._compute_derivatives(
            x_bc_pos, up_to_order=3,
        )

        derivs = {0: w_bc, 1: w_bc_x, 2: w_bc_xx, 3: w_bc_xxx}
        loss_BC = self.pinn_losses.loss_function_BC(
            derivs=derivs,
            bc_orders=bc_orders,
            y_true=y_train_BC,
        ) * self.hparams.loss_BC_param

        loss = loss_PDE + loss_BC

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
        w_pred = self.forward(pred_batch[0])
        return w_pred

    def compute_pde_residual(self, x_candidates: torch.Tensor) -> torch.Tensor:
        """Compute absolute PDE residual for adaptive refinement (RAR callback).

        Args:
            x_candidates: Candidate points, shape (n, 1).

        Returns:
            Absolute residual |EI * w_xxxx - q|, shape (n, 1).
        """
        x = x_candidates[:, 0:1].clone().detach().requires_grad_(True)
        _, _, _, _, w_xxxx = self._compute_derivatives(x, up_to_order=4)
        residual = self.hparams.EI * w_xxxx - self.hparams.q
        return residual.abs().detach()
