""" PINN for thick-walled cylinder (Lame problem).

Governing ODE (Euler-Cauchy equation for radial displacement):
    d2u/dr2 + (1/r) du/dr - u/r2 = 0

Single-output network: input r -> output u(r).
Stresses are derived via automatic differentiation and Hooke's law.

Boundary conditions are on sigma_r (radial stress), computed from u and du/dr:
    sigma_r = E/(1-nu^2) * (du/dr + nu * u/r)

References:
    Lame, G. (1852).
        "Lecons sur la theorie mathematique de l'elasticite des corps solides."

    Timoshenko, S.P. & Goodier, J.N. (1970).
        "Theory of Elasticity." McGraw-Hill, 3rd edition.

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


class CylinderPINNLosses:
    def __init__(self) -> None:
        pass

    @staticmethod
    def loss_function_PDE(
        u: torch.Tensor,
        u_r: torch.Tensor,
        u_rr: torch.Tensor,
        r: torch.Tensor,
    ) -> torch.Tensor:
        """PDE residual: d2u/dr2 + (1/r)*du/dr - u/r^2 = 0."""
        residual = u_rr + u_r / r - u / r**2
        return torch.mean(residual ** 2)

    @staticmethod
    def loss_function_BC(
        sigma_r_pred: torch.Tensor,
        sigma_r_target: torch.Tensor,
    ) -> torch.Tensor:
        """BC loss: prescribed radial stress at inner/outer surfaces."""
        return torch.mean((sigma_r_pred - sigma_r_target) ** 2)


class CylinderPINNRegressor(pl.LightningModule):
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

        self.pinn_losses = CylinderPINNLosses()

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

    def net(self, r):
        """Forward pass with radial input for autograd."""
        return self.forward(r)

    def _compute_derivatives(self, r):
        """Compute u, du/dr, d2u/dr2 from the network.

        Args:
            r: Input tensor with requires_grad=True, shape (n, 1).

        Returns:
            (u, u_r, u_rr): Displacement and its derivatives.
        """
        ones = torch.ones_like(r)
        u = self.net(r)

        u_r = torch.autograd.grad(
            outputs=u, inputs=r,
            grad_outputs=ones,
            retain_graph=True, create_graph=True,
        )[0]

        u_rr = torch.autograd.grad(
            outputs=u_r, inputs=r,
            grad_outputs=ones,
            retain_graph=True, create_graph=True,
        )[0]

        return u, u_r, u_rr

    def _compute_sigma_r(self, u, u_r, r):
        """Compute radial stress from displacement and its derivative.

        sigma_r = E/(1-nu^2) * (du/dr + nu * u/r)
        """
        E = self.hparams.E
        nu = self.hparams.nu_poisson
        return (E / (1.0 - nu**2)) * (u_r + nu * u / r)

    def _shared_eval_step(self, eval_batch, eval_batch_idx):
        pass

    def training_step(self, train_batch, batch_idx) -> torch.Tensor:
        # * Part 1: Unpack batch
        r_train_PDE = train_batch[0][0]
        x_train_BC, y_train_BC = train_batch[1]

        total_BC = int(torch.tensor(len(x_train_BC)))
        total_PDE = int(torch.tensor(len(r_train_PDE)))

        # * Part 2: PDE loss
        r_pde = r_train_PDE[:, 0:1].clone().detach().requires_grad_(True)
        u, u_r, u_rr = self._compute_derivatives(r_pde)

        loss_PDE = self.pinn_losses.loss_function_PDE(
            u=u, u_r=u_r, u_rr=u_rr, r=r_pde,
        ) * self.hparams.loss_PDE_param

        # * Part 3: BC loss (stress-based)
        r_bc = x_train_BC[:, 0:1].clone().detach().requires_grad_(True)
        u_bc, u_r_bc, _ = self._compute_derivatives(r_bc)

        sigma_r_pred = self._compute_sigma_r(u_bc, u_r_bc, r_bc)

        loss_BC = self.pinn_losses.loss_function_BC(
            sigma_r_pred=sigma_r_pred,
            sigma_r_target=y_train_BC,
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
        u_pred = self.forward(pred_batch[0])
        return u_pred

    def compute_pde_residual(self, r_candidates: torch.Tensor) -> torch.Tensor:
        """Compute absolute PDE residual for adaptive refinement (RAR callback).

        Args:
            r_candidates: Candidate points, shape (n, 1).

        Returns:
            Absolute residual, shape (n, 1).
        """
        r = r_candidates[:, 0:1].clone().detach().requires_grad_(True)
        u, u_r, u_rr = self._compute_derivatives(r)
        residual = u_rr + u_r / r - u / r**2
        return residual.abs().detach()
