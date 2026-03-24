""" Generate dataset for Raissi's Burgers equation.

Data source: burgers_shock.mat from Raissi et al. (2019).
    https://github.com/maziarraissi/PINNs

Reference:
    Raissi, M., Perdikaris, P., & Karniadakis, G.E. (2019).
        "Physics-informed neural networks: A deep learning framework for
        solving forward and inverse problems involving nonlinear partial
        differential equations." Journal of Computational Physics, 378, 686-707.
        https://doi.org/10.1016/j.jcp.2018.10.045
"""
import os

import numpy as np
import scipy

from pyDOE import lhs

L_REF = 8.0  # Reference length for non-dimensionalization (half-domain width)


def generate_dataset(path: str = "./src/raissi_burgers/data/",):
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

    # Initial condition
    x_train_IC = np.hstack((X[0:1, :].T, T[0:1, :].T))     # (256, 2)
    u_exact_IC = u_exact[0:1, :].T                           # (256, 1)

    # Boundary condition
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

    # * Non-dimensionalization: scale x to [-1, 1]
    all_x_train_IC_BC[:, 0] /= L_REF
    x_train[:, 0] /= L_REF
    x_star[:, 0] /= L_REF

    # * Final data
    idx = np.random.choice(all_x_train_IC_BC.shape[0], n_bc_points, replace=False)
    x_train_IC_BC = all_x_train_IC_BC[idx, :]   # (100, 2)
    y_train_IC_BC = all_y_train_IC_BC[idx, :]    # (100, 1)

    # * Save
    np.save(os.path.join(path, "x_train_IC_BC"), x_train_IC_BC)
    np.save(os.path.join(path, "y_train_IC_BC"), y_train_IC_BC)
    np.save(os.path.join(path, "x_train"), x_train)
    np.save(os.path.join(path, "x_star"), x_star)
    np.save(os.path.join(path, "scaling"), {"L_ref": L_REF})

if __name__ == "__main__":
    generate_dataset()
