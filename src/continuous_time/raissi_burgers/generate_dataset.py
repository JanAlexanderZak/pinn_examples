""" Generate Moseley's dataset.

Dictionary:
nu
N_u
N_f: collocation_points
t: t_domain
x
X
T
Exact: y_true
X_star
u_star
lb: lower_boundary
up: upper_boundary
xx1
uu1

comments indicate matrix shape
"""
import os

import numpy as np
import scipy

from pyDOE import lhs


def generate_dataset(path: str = "./src/continuous_time/raissi_burgers/data/",):
    N_u = 100
    collocation_points = 10000
    
    data = scipy.io.loadmat("src/continuous_time/raissi_burgers/data/burgers_shock.mat")
    t_domain = data["t"].flatten().reshape(-1, 1)  # (100, 1)
    x_domain = data["x"].flatten().reshape(-1, 1)  # (256, 1)
    y_true = np.real(data["usol"]).T

    X, T = np.meshgrid(x_domain, t_domain)  # (100, 256), (100, 256)
    y_true = np.hstack((
        X.flatten().reshape(-1, 1),
        T.flatten().reshape(-1, 1),
    ))  # (256*100=25600, 1)
    y_true_flat = y_true.flatten().reshape(-1, 1)  # (256*100=25600, 1)

    # * Boundary conditions
    lower_boundary = y_true.min(axis=0)
    upper_boundary = y_true.max(axis=0)

    # Initial condition
    x_train_IC = np.hstack((X[0:1, :].T, T[0:1, :].T))     # (256, 2)
    y_train_IC = y_true[0:1, :].T                          # (256, 1)

    # Boundary condition
    x_train_BC_lb = np.hstack((X[:, 0:1], T[:, 0:1]))         # (100, 2)
    y_train_BC_lb = y_true[:, 0:1]                            # (100, 1)
    x_train_BC_ub = np.hstack((X[:, -1:], T[:, -1:]))         # (100, 2)
    y_train_BC_ub = y_true[:, -1:]                            # (100, 1)

    all_x_train_BC = np.vstack([x_train_IC, x_train_BC_lb, x_train_BC_ub])
    all_y_train_BC = np.vstack([y_train_IC, y_train_BC_lb, y_train_BC_ub])

    # * Sample collocation
    x_train = lower_boundary + (upper_boundary - lower_boundary) * lhs(2, collocation_points)
    x_train = np.vstack((x_train, all_x_train_BC))  # (10000+456=10456, 2)
    
    # * Final data
    idx = np.random.choice(all_x_train_BC.shape[0], N_u, replace=False)
    x_train_BC = all_x_train_BC[idx, :]   # (100, 2)
    y_train_BC = all_y_train_BC[idx, :]       # (100, 1)

    # * Save
    np.save(os.path.join(path, "X_u_train"), x_train_BC)
    np.save(os.path.join(path, "u_train"), y_train_BC)
    np.save(os.path.join(path, "X_f_train"), x_train)
    np.save(os.path.join(path, "X_star"), y_true)

if __name__ == "__main__":
    generate_dataset()
