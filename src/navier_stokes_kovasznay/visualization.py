import torch
import numpy as np

from scipy.interpolate import griddata

from src.navier_stokes_kovasznay.generate_dataset import exact_solution, scale_coords
from src.visualization import plot_heatmap_comparison


def main(epoch, save_dir="src/figures"):
    Re = 20.0
    nu = 1.0 / Re

    x_resolution = 80
    y_resolution = 80
    x_domain = np.linspace(-0.5, 1.0, x_resolution)
    y_domain = np.linspace(-0.5, 1.5, y_resolution)
    X, Y = np.meshgrid(x_domain, y_domain)

    u_exact, v_exact, p_exact = exact_solution(X, Y, nu)

    # Use scaled coordinates for interpolation (matching PINN input space)
    X_s, Y_s = scale_coords(X, Y)
    x_test_grid = np.hstack((
        X_s.flatten().reshape(-1, 1),
        Y_s.flatten().reshape(-1, 1),
    ))

    # Load predictions
    uvp_pred_raw = torch.load(
        f"./src/navier_stokes_kovasznay/data/predictions/predictions_{epoch}.pkl"
    )
    uvp_pred_raw = torch.cat(uvp_pred_raw, dim=0).numpy()

    u_pred = griddata(x_test_grid, uvp_pred_raw[:, 0], (X_s, Y_s), method="cubic")
    v_pred = griddata(x_test_grid, uvp_pred_raw[:, 1], (X_s, Y_s), method="cubic")
    p_pred = griddata(x_test_grid, uvp_pred_raw[:, 2], (X_s, Y_s), method="cubic")

    # Plot: exact vs PINN vs difference for each field
    fields = [
        ("u_velocity", r"$u$ [-]", "Kovasznay Flow — $u$ velocity", u_exact, u_pred),
        ("v_velocity", r"$v$ [-]", "Kovasznay Flow — $v$ velocity", v_exact, v_pred),
        ("pressure_p", r"$p$ [-]", "Kovasznay Flow — pressure $p$", p_exact, p_pred),
    ]
    for safe_name, field_label, title, exact, pred in fields:
        plot_heatmap_comparison(
            exact=exact,
            pred=pred,
            x_coords=x_domain,
            y_coords=y_domain,
            x_label=r"$x$ [-]",
            y_label=r"$y$ [-]",
            field_label=field_label,
            save_path=f"{save_dir}/navier_stokes_kovasznay_{safe_name}_comparison.png",
            title=title,
        )


if __name__ == "__main__":
    main(15000)
