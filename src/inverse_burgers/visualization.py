import torch
import numpy as np
import scipy

from scipy.interpolate import griddata

from src.visualization import plot_heatmap_comparison, plot_line


def main(epoch, save_dir="src/figures"):
    # Load reference data
    data = scipy.io.loadmat("src/raissi_burgers/data/burgers_shock.mat")
    t_domain = data["t"].flatten()
    x_domain = data["x"].flatten()
    u_exact = np.real(data["usol"]).T

    X, T = np.meshgrid(x_domain, t_domain)

    x_test_grid = np.hstack((
        X.flatten().reshape(-1, 1),
        T.flatten().reshape(-1, 1),
    ))

    # * Plot 1: Solution comparison (analytical vs PINN vs difference)
    pred_dir = (
        "./src/inverse_burgers/data/predictions"
    )
    u_pred_raw = torch.load(
        f"{pred_dir}/predictions_{epoch}.pkl",
    )
    u_pred_raw = torch.tensor(u_pred_raw).reshape(-1, 1)
    u_pred = griddata(x_test_grid, u_pred_raw.flatten(), (X, T), method="cubic")

    plot_heatmap_comparison(
        exact=u_exact.T,
        pred=u_pred.T,
        x_coords=t_domain,
        y_coords=x_domain,
        x_label="t",
        y_label="x",
        field_label="u(x,t)",
        save_path=f"{save_dir}/inverse_burgers_solution_comparison.png",
    )

    # * Plot 2: nu convergence
    nus = torch.load(f"{pred_dir}/nus_{epoch}.pkl")
    nu_true = 0.01 / np.pi

    plot_line(
        x=range(len(nus)),
        ys=[nus],
        labels=["Learned nu"],
        title="Viscosity Parameter Convergence",
        xlabel="Training Step",
        ylabel="nu",
        save_path=f"{save_dir}/inverse_burgers_nu_convergence.png",
        hlines=[(nu_true, f"True nu = {nu_true:.6f}", "red")],
    )


if __name__ == "__main__":
    main(20500)
