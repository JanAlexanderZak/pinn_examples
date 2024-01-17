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
    X_star = np.hstack((
        X.flatten().reshape(-1, 1),
        T.flatten().reshape(-1, 1),
    ))  # (256*100=25600, 1)
    y_true_flat = y_true.flatten().reshape(-1, 1)  # (256*100=25600, 1)

    # * Boundary conditions
    lower_boundary = X_star.min(axis=0)
    upper_boundary = X_star.max(axis=0)

    xx1 = np.hstack((X[0:1, :].T, T[0:1, :].T))     # (256, 2)
    uu1 = y_true[0:1, :].T                          # (256, 1)
    xx2 = np.hstack((X[:, 0:1], T[:, 0:1]))         # (100, 2)
    uu2 = y_true[:, 0:1]                            # (100, 1)
    xx3 = np.hstack((X[:, -1:], T[:, -1:]))         # (100, 2)
    uu3 = y_true[:, -1:]                            # (100, 1)

    all_X_u_train = np.vstack([xx1, xx2, xx3])
    all_u_train = np.vstack([uu1, uu2, uu3])

    # * Sample collocation
    X_f_train = lower_boundary + (upper_boundary - lower_boundary) * lhs(2, collocation_points)
    X_f_train = np.vstack((X_f_train, all_X_u_train))  # (10000+456=10456, 2)
    
    # * Final data
    idx = np.random.choice(all_X_u_train.shape[0], N_u, replace=False)
    X_u_train = all_X_u_train[idx, :]   # (100, 2)
    u_train = all_u_train[idx, :]       # (100, 1)

    # * Save
    np.save(os.path.join(path, "X_u_train"), X_u_train)
    np.save(os.path.join(path, "u_train"), u_train)
    np.save(os.path.join(path, "X_f_train"), X_f_train)
    np.save(os.path.join(path, "X_star"), X_star)

if __name__ == "__main__":
    generate_dataset()
