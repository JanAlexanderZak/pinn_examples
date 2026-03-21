""" Generate dataset for inverse Burgers equation.

Same PDE as Raissi's Burgers: u_t + u*u_x = nu*u_xx
But here nu is unknown and must be identified from sparse noisy observations.
True value: nu = 0.01/pi
"""
import os

import numpy as np
import scipy

from pyDOE import lhs


def generate_dataset(path: str = "./src/inverse_burgers/data/"):
    n_bc_points = 100
    n_collocation_points = 10000
    n_observation_points = 200
    noise_level = 0.01

    data = scipy.io.loadmat("src/raissi_burgers/data/burgers_shock.mat")
    t_domain = data["t"].flatten().reshape(-1, 1)  # (100, 1)
    x_domain = data["x"].flatten().reshape(-1, 1)  # (256, 1)
    u_exact = np.real(data["usol"]).T

    X, T = np.meshgrid(x_domain, t_domain)  # (100, 256), (100, 256)
    x_star = np.hstack((
        X.flatten().reshape(-1, 1),
        T.flatten().reshape(-1, 1),
    ))  # (25600, 2)

    # * Boundary conditions
    lower_boundary = x_star.min(axis=0)
    upper_boundary = x_star.max(axis=0)

    # Initial condition
    x_train_IC = np.hstack((X[0:1, :].T, T[0:1, :].T))
    u_exact_IC = u_exact[0:1, :].T

    # Boundary condition
    x_train_BC_lb = np.hstack((X[:, 0:1], T[:, 0:1]))
    u_exact_BC_lb = u_exact[:, 0:1]
    x_train_BC_ub = np.hstack((X[:, -1:], T[:, -1:]))
    u_exact_BC_ub = u_exact[:, -1:]

    all_x_train_IC_BC = np.vstack([x_train_IC, x_train_BC_lb, x_train_BC_ub])
    all_y_train_IC_BC = np.vstack([u_exact_IC, u_exact_BC_lb, u_exact_BC_ub])

    # * Sample collocation
    x_train = (
        lower_boundary
        + (upper_boundary - lower_boundary)
        * lhs(2, n_collocation_points)
    )
    x_train = np.vstack((x_train, all_x_train_IC_BC))

    # * Subsample IC/BC
    idx = np.random.choice(all_x_train_IC_BC.shape[0], n_bc_points, replace=False)
    x_train_IC_BC = all_x_train_IC_BC[idx, :]
    y_train_IC_BC = all_y_train_IC_BC[idx, :]

    # * Sparse noisy observations from the interior domain
    u_exact_flat = u_exact.flatten().reshape(-1, 1)  # (25600, 1)
    obs_idx = np.random.choice(x_star.shape[0], n_observation_points, replace=False)
    x_obs = x_star[obs_idx, :]
    u_obs = (
        u_exact_flat[obs_idx, :]
        + noise_level * np.random.randn(n_observation_points, 1)
    )

    # * Save
    np.save(os.path.join(path, "x_train_IC_BC"), x_train_IC_BC)
    np.save(os.path.join(path, "y_train_IC_BC"), y_train_IC_BC)
    np.save(os.path.join(path, "x_train"), x_train)
    np.save(os.path.join(path, "x_star"), x_star)
    np.save(os.path.join(path, "x_obs"), x_obs)
    np.save(os.path.join(path, "u_obs"), u_obs)


if __name__ == "__main__":
    generate_dataset()
