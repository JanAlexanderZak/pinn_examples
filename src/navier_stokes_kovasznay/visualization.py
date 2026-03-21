import torch
import numpy as np

from scipy.interpolate import griddata

from src.navier_stokes_kovasznay.generate_dataset import exact_solution
from src.visualization import plot_heatmap_comparison


def main(epoch):
    Re = 20.0
    nu = 1.0 / Re

    x_resolution = 80
    y_resolution = 80
    x_domain = np.linspace(-0.5, 1.0, x_resolution)
    y_domain = np.linspace(-0.5, 1.5, y_resolution)
    X, Y = np.meshgrid(x_domain, y_domain)

    u_exact, v_exact, p_exact = exact_solution(X, Y, nu)

    x_test_grid = np.hstack((
        X.flatten().reshape(-1, 1),
        Y.flatten().reshape(-1, 1),
    ))

    # Load predictions
    uvp_pred_raw = torch.load(
        f"./src/navier_stokes_kovasznay/data/predictions/predictions_{epoch}.pkl"
    )
    uvp_pred_raw = torch.cat(uvp_pred_raw, dim=0).numpy()

    u_pred = griddata(x_test_grid, uvp_pred_raw[:, 0], (X, Y), method="cubic")
    v_pred = griddata(x_test_grid, uvp_pred_raw[:, 1], (X, Y), method="cubic")
    p_pred = griddata(x_test_grid, uvp_pred_raw[:, 2], (X, Y), method="cubic")

    # Plot: exact vs PINN vs difference for each field
    for field_name, exact, pred in [("u velocity", u_exact, u_pred),
                                     ("v velocity", v_exact, v_pred),
                                     ("pressure p", p_exact, p_pred)]:
        safe_name = field_name.replace(" ", "_")
        plot_heatmap_comparison(
            exact=exact,
            pred=pred,
            x_coords=x_domain,
            y_coords=y_domain,
            x_label="x",
            y_label="y",
            field_label=field_name,
            save_path=f"src/navier_stokes_kovasznay/plots/{safe_name}_comparison.png",
            title=f"Kovasznay Flow - {field_name}",
        )


if __name__ == "__main__":
    main(15000)
