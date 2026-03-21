import os

import numpy as np
import pytorch_lightning as pl
import torch

from pyDOE import lhs


class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.

    Reference:
        https://github.com/Lightning-AI/lightning/issues/2534#issuecomment-674582085

    Example:
        Trainer(callbacks=[CheckpointEveryNSteps()])
    """

    def __init__(
        self,
        save_step_frequency,
        prefix="N-Step-Checkpoint",
        use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    def on_train_epoch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}_{epoch=}_{global_step=}.ckpt"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)


class ResidualAdaptiveRefinement(pl.Callback):
    """ Residual-Based Adaptive Refinement (RAR) callback.

    Periodically evaluates PDE residuals on a dense candidate grid and adds
    the highest-residual points to the collocation set. This focuses training
    on regions where the PDE is least satisfied (e.g., near shocks, sharp gradients).

    Requirements:
        - The model must implement a `compute_pde_residual(x_candidates)` method
          that returns per-point PDE residual magnitudes as a 1D tensor.
        - The data module must implement an `update_collocation_points(new_points)`
          method that adds points to the collocation dataset.

    Example:
        rar = ResidualAdaptiveRefinement(
            lower_boundary=[-1.0, 0.0],
            upper_boundary=[1.0, 1.0],
            n_candidates=10000,
            n_add=100,
            refinement_interval=500,
        )
        trainer = pl.Trainer(callbacks=[rar])
    """

    def __init__(
        self,
        lower_boundary,
        upper_boundary,
        n_candidates: int = 10000,
        n_add: int = 100,
        refinement_interval: int = 500,
    ):
        """
        Args:
            lower_boundary: Lower bounds of the domain (list or array).
            upper_boundary: Upper bounds of the domain (list or array).
            n_candidates: Number of candidate points to evaluate residuals on.
            n_add: Number of highest-residual points to add each refinement step.
            refinement_interval: Refine every N epochs.
        """
        super().__init__()
        self.lower_boundary = np.array(lower_boundary, dtype=np.float32)
        self.upper_boundary = np.array(upper_boundary, dtype=np.float32)
        self.n_dim = len(self.lower_boundary)
        self.n_candidates = n_candidates
        self.n_add = n_add
        self.refinement_interval = refinement_interval

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        epoch = trainer.current_epoch
        if epoch == 0 or epoch % self.refinement_interval != 0:
            return

        if not hasattr(pl_module, "compute_pde_residual"):
            return

        # Generate dense candidate points via LHS
        candidates = (
            self.lower_boundary
            + (self.upper_boundary - self.lower_boundary)
            * lhs(self.n_dim, self.n_candidates)
        )
        candidates_tensor = torch.tensor(
            candidates, dtype=torch.float32,
        ).to(pl_module.device)

        # Evaluate PDE residuals
        pl_module.eval()
        with torch.no_grad():
            residuals = pl_module.compute_pde_residual(candidates_tensor)
        pl_module.train()

        # Select top-k highest residual points
        _, top_idx = torch.topk(residuals.flatten(), min(self.n_add, len(residuals)))
        new_points = candidates[top_idx.cpu().numpy()]

        # Update the data module's collocation set
        data_module = trainer.datamodule
        has_method = hasattr(data_module, "update_collocation_points")
        if data_module is not None and has_method:
            data_module.update_collocation_points(new_points)
            # Recreate the train dataloader with updated data
            trainer.reset_train_dataloader()

            if epoch % (self.refinement_interval * 2) == 0:
                print(
                    f"[RAR] Epoch {epoch}: Added {len(new_points)} points."
                    f" Max residual: {residuals.max().item():.5e},"
                    f" Mean residual: {residuals.mean().item():.5e}"
                )
