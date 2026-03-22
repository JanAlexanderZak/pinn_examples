import torch
import numpy as np

from scipy.interpolate import griddata

from src.wave_eq_1d.generate_dataset import exact_solution
from src.visualization import plot_heatmap_comparison


def main(epoch, save_dir="src/figures"):
    # Repeat dataset generation
    domain_length = 1.0
    x_resolution = 51

    x_domain = np.linspace(0., domain_length, x_resolution)
    t_domain = np.linspace(0, 2.0, 101)

    # * Initial and boundary conditions
    X, T = np.meshgrid(x_domain, t_domain)

    x_test_grid = np.hstack((
        X.flatten().reshape(-1, 1),
        T.flatten().reshape(-1, 1),
    ))

    u_analytical = np.vstack([
        exact_solution(x=x_domain, t=time, c=1.0)
        for time in t_domain
    ])

    # * Prediction
    pred_path = (
        "./src/wave_eq_1d"
        f"/data/predictions/predictions_{epoch}.pkl"
    )
    u_pred_raw = torch.load(pred_path)
    u_pred_raw = torch.tensor(u_pred_raw).reshape(-1, 1)
    u_pred = griddata(x_test_grid, u_pred_raw.flatten(), (X, T), method="cubic")

    plot_heatmap_comparison(
        exact=u_analytical.T,
        pred=u_pred.T,
        x_coords=t_domain,
        y_coords=x_domain,
        x_label="t",
        y_label="x",
        field_label="u(x,t)",
        save_path=f"{save_dir}/wave_eq_1d_analytical_vs_pinn.png",
    )


if __name__ == "__main__":
    main(10000)
