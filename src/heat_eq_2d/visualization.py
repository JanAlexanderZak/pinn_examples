import torch
import numpy as np

from scipy.interpolate import griddata

from src.heat_eq_2d.generate_dataset import (
    fdm_solution, PLATE_LENGTH, MAX_ITER_TIME, ALPHA, DOMAIN_LENGTH, U_REF,
)
from src.visualization import plot_heatmap_comparison


def main(epoch, save_dir="src/figures"):
    n_points = PLATE_LENGTH
    n_time_steps = MAX_ITER_TIME

    x_resolution = n_points
    y_resolution = n_points
    t_resolution = n_time_steps

    # FDM reference solution and physical time
    y_fdm, t_physical = fdm_solution(
        n_points=n_points,
        n_time_steps=n_time_steps,
        alpha=ALPHA,
        domain_length=DOMAIN_LENGTH,
    )

    x_domain = np.linspace(0, DOMAIN_LENGTH, x_resolution)
    y_domain = np.linspace(0, DOMAIN_LENGTH, y_resolution)
    t_domain = np.linspace(0, t_physical, t_resolution)

    X, Y, T = np.meshgrid(x_domain, y_domain, t_domain, indexing="ij")
    x_star = np.hstack((
        X.reshape(-1, 1), Y.reshape(-1, 1), T.reshape(-1, 1),
    ))

    # Load PINN predictions
    u_pred_raw = torch.load(
        f"./src/heat_eq_2d/data/predictions/predictions_{epoch}.pkl"
    )
    u_pred_raw = torch.tensor(u_pred_raw).reshape(-1, 1) * U_REF

    # Compare at the final time step
    t_idx = -1
    X_2d, Y_2d = np.meshgrid(x_domain, y_domain, indexing="ij")

    # Exact at final time
    exact_final = y_fdm[t_idx, :, :]  # (n_points, n_points)
    # FDM grid maps to [0, DOMAIN_LENGTH]
    x_fdm = np.linspace(0, DOMAIN_LENGTH, y_fdm.shape[1])
    y_fdm_coords = np.linspace(0, DOMAIN_LENGTH, y_fdm.shape[2])
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
        x_label=r"$x$ [-]",
        y_label=r"$y$ [-]",
        field_label=r"$T$ [-]",
        save_path=f"{save_dir}/heat_eq_2d_analytical_vs_pinn.png",
        title=f"2D Heat Equation at t={t_domain[t_idx]:.6f}",
    )


if __name__ == "__main__":
    main(20002)
