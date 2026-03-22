import torch
import scipy
import numpy as np

from pyDOE import lhs
from scipy.interpolate import griddata

from src.visualization import plot_single_heatmap


def main(epoch, save_dir="src/figures"):

    np.random.seed(6020)

    n_bc_points = 100
    n_collocation_points = 10000

    data = scipy.io.loadmat("src/raissi_burgers/data/burgers_shock.mat")
    t_domain = data["t"].flatten().reshape(-1, 1)  # (100, 1)
    x_domain = data["x"].flatten().reshape(-1, 1)  # (256, 1)
    u_exact = np.real(data["usol"]).T

    X, T = np.meshgrid(x_domain, t_domain)  # (100, 256), (100, 256)
    x_star = np.hstack((
        X.flatten().reshape(-1, 1),
        T.flatten().reshape(-1, 1),
    ))  # (256*100=25600, 2)

    # * Boundary conditions
    lower_boundary = x_star.min(axis=0)
    upper_boundary = x_star.max(axis=0)

    x_train_IC = np.hstack((X[0:1, :].T, T[0:1, :].T))     # (256, 2)
    u_exact_IC = u_exact[0:1, :].T                           # (256, 1)
    x_train_BC_lb = np.hstack((X[:, 0:1], T[:, 0:1]))       # (100, 2)
    u_exact_BC_lb = u_exact[:, 0:1]                          # (100, 1)
    x_train_BC_ub = np.hstack((X[:, -1:], T[:, -1:]))       # (100, 2)
    u_exact_BC_ub = u_exact[:, -1:]                          # (100, 1)

    all_x_train_IC_BC = np.vstack([x_train_IC, x_train_BC_lb, x_train_BC_ub])
    all_y_train_IC_BC = np.vstack([u_exact_IC, u_exact_BC_lb, u_exact_BC_ub])

    # * Sample collocation
    x_train = (
        lower_boundary
        + (upper_boundary - lower_boundary)
        * lhs(2, n_collocation_points)
    )
    x_train = np.vstack((x_train, all_x_train_IC_BC))  # (10000+456=10456, 2)

    # * Final data
    idx = np.random.choice(all_x_train_IC_BC.shape[0], n_bc_points, replace=False)
    x_train_IC_BC = all_x_train_IC_BC[idx, :]   # (100, 2)
    y_train_IC_BC = all_y_train_IC_BC[idx, :]    # (100, 1)

    # Predictions
    path = (
        "./src/raissi_burgers"
        f"/data/predictions/predictions_{epoch}.pkl"
    )
    u_pred_raw = torch.load(path)
    u_pred_raw = torch.tensor(u_pred_raw).reshape(-1, 1)
    u_pred = griddata(x_star, u_pred_raw.flatten(), (X, T), method='cubic')

    # * Visualize
    plot_single_heatmap(
        data=u_pred.T,
        extent=[
            t_domain.min(),
            t_domain.max(),
            x_domain.min(),
            x_domain.max(),
        ],
        title="$u(x,t)$",
        x_label="$t$",
        y_label="$x$",
        cbar_label="u",
        save_path=f"{save_dir}/raissi_burgers_solution.png",
        scatter_data=(x_train_IC_BC[:, 1], x_train_IC_BC[:, 0]),
        scatter_kwargs={
            "marker": "x",
            "color": "k",
            "label": f"Data ({y_train_IC_BC.shape[0]} points)",
            "markersize": 4,
            "linestyle": "None",
            "alpha": 1.0,
        },
        vlines=[
            t_domain[25].item(),
            t_domain[50].item(),
            t_domain[75].item(),
        ],
    )

if __name__ == "__main__":
    main(20500)
