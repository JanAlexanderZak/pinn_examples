import torch
import numpy as np

from scipy.interpolate import griddata

from src.heat_eq_2d.generate_dataset import (
    fdm_solution, PLATE_LENGTH, MAX_ITER_TIME, ALPHA,
)
from src.visualization import plot_heatmap_comparison


def main(epoch, save_dir="src/figures"):
    plate_length = PLATE_LENGTH
    max_iter_time = MAX_ITER_TIME

    x_resolution = 50
    y_resolution = 50
    t_resolution = 50

    x_domain = np.linspace(0, plate_length, x_resolution)
    y_domain = np.linspace(0, plate_length, y_resolution)
    t_domain = np.linspace(0, max_iter_time, t_resolution)

    X, Y, T = np.meshgrid(x_domain, y_domain, t_domain, indexing="ij")
    x_star = np.hstack((
        X.reshape(-1, 1), Y.reshape(-1, 1), T.reshape(-1, 1),
    ))

    # FDM reference solution: shape (max_iter_time, plate_length, plate_length)
    y_fdm = fdm_solution(plate_length, max_iter_time, ALPHA)

    # Load PINN predictions
    u_pred_raw = torch.load(
        f"./src/heat_eq_2d/data/predictions/predictions_{epoch}.pkl"
    )
    u_pred_raw = torch.tensor(u_pred_raw).reshape(-1, 1)

    # Compare at the final time step
    t_idx = -1
    X_2d, Y_2d = np.meshgrid(x_domain, y_domain, indexing="ij")

    # Exact at final time
    exact_final = y_fdm[t_idx, :, :]  # (plate_length, plate_length)
    # Interpolate exact to our grid resolution
    x_fdm = np.linspace(0, plate_length, y_fdm.shape[1])
    y_fdm_coords = np.linspace(0, plate_length, y_fdm.shape[2])
    X_fdm, Y_fdm = np.meshgrid(x_fdm, y_fdm_coords, indexing="ij")
    exact_interp = griddata(
        np.hstack((X_fdm.reshape(-1, 1), Y_fdm.reshape(-1, 1))),
        exact_final.flatten(),
        (X_2d, Y_2d),
        method="cubic",
    )

    # PINN prediction at final time
    mask = np.isclose(x_star[:, 2], t_domain[t_idx])
    x_final = x_star[mask, :2]
    u_final = u_pred_raw.numpy().flatten()[mask]
    pred_interp = griddata(x_final, u_final, (X_2d, Y_2d), method="cubic")

    plot_heatmap_comparison(
        exact=exact_interp,
        pred=pred_interp,
        x_coords=x_domain,
        y_coords=y_domain,
        x_label="x",
        y_label="y",
        field_label="T(x,y)",
        save_path=f"{save_dir}/heat_eq_2d_analytical_vs_pinn.png",
        title=f"2D Heat Equation at t={t_domain[t_idx]:.1f}",
    )


if __name__ == "__main__":
    main(20002)
