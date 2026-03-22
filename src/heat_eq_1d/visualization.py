import torch
import numpy as np

from scipy.interpolate import griddata

from src.heat_eq_1d.generate_dataset import exact_solution
from src.visualization import plot_heatmap_comparison


def main(epoch, save_dir="src/figures"):
    # Repeat dataset generation
    domain_length = 1.
    x_resolution = 21

    x_domain = np.linspace(0., domain_length, x_resolution)
    t_domain = np.linspace(0, 5, 100)

    # * Initial and boundary conditions
    X, T = np.meshgrid(x_domain, t_domain)  # (100, 21), (100, 21)

    x_test_grid = np.hstack((
        X.flatten().reshape(-1, 1),
        T.flatten().reshape(-1, 1),
    ))

    u_analytical = np.vstack([
        exact_solution(x=x_domain, t=time, alpha=0.1)
        for time in t_domain
    ])

    # * Prediction
    u_pred_raw = torch.load(
        f"./src/heat_eq_1d/data/"
        f"predictions/predictions_{epoch}.pkl"
    )
    u_pred_raw = torch.tensor(u_pred_raw).reshape(-1, 1)
    u_pred = griddata(x_test_grid, u_pred_raw.flatten(), (X, T), method="cubic")

    plot_heatmap_comparison(
        exact=u_analytical.T,
        pred=u_pred.T,
        x_coords=t_domain,
        y_coords=x_domain,
        x_label="t",
        y_label="x",
        field_label="T(x,t)",
        save_path=f"{save_dir}/heat_eq_1d_analytical_vs_pinn.png",
    )


if __name__ == "__main__":
    main(10000)
