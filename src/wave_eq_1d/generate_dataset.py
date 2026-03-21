"""
1D Wave Equation PINN dataset generation.

PDE: u_tt = c^2 * u_xx
Domain: x in [0, L], t in [0, T]
BC: u(0, t) = u(L, t) = 0  (fixed endpoints)
IC: u(x, 0) = sin(pi * x / L)  (displacement)
    u_t(x, 0) = 0              (zero initial velocity)

Analytical solution: u(x, t) = sin(pi * x / L) * cos(pi * c * t / L)
"""
import os

from typing import List

import numpy as np
import matplotlib.pyplot as plt

from pyDOE import lhs

from src.visualization import pinn_style, save_figure, CMAP_SEQUENTIAL


def exact_solution(
    x: List[float],
    t: float,
    c: float,
    L: float = 1.0,
):
    """ Returns the exact solution of the 1D wave equation.

    Args:
        x (List[float]): spatial domain
        t (float): time
        c (float): wave speed
        L (float): domain length

    Returns:
        displacement u(x, t)
    """
    return np.sin(np.pi * x / L) * np.cos(np.pi * c * t / L)


def generate_dataset(path: str = "src/wave_eq_1d/data"):
    # Physical parameters
    c = 1.0                                         # Wave speed
    domain_length = 1.0                             # Size of computational domain

    # Grid parameters
    x_resolution = 51                               # number of grid points
    x_domain = np.linspace(0., domain_length, x_resolution)

    t_end = 2.0
    t_resolution = 101
    time_domain = np.linspace(0, t_end, t_resolution)

    u_exact = np.vstack([
        exact_solution(x=x_domain, t=time, c=c)
        for time in time_domain
    ])

    with pinn_style():
        fig, ax = plt.subplots()
        im = ax.imshow(u_exact, origin="lower", cmap=CMAP_SEQUENTIAL, aspect="auto")
        ax.set_xlabel("t")
        ax.set_ylabel("x")
        fig.colorbar(im, ax=ax)
        save_figure(fig, "src/wave_eq_1d/plots/analytical_solution.png")

    # * Initial and boundary conditions
    X, T = np.meshgrid(x_domain, time_domain)  # (t_resolution, x_resolution)

    x_star = np.hstack((
        X.flatten().reshape(-1, 1),
        T.flatten().reshape(-1, 1),
    ))

    lower_boundary = x_star.min(axis=0)
    upper_boundary = x_star.max(axis=0)

    # Initial conditions: u(x, 0) = sin(pi*x)
    x_train_IC = np.hstack((X[0:1, :].T, T[0:1, :].T))
    u_exact_IC = u_exact[0, :].reshape(-1, 1)

    # Boundary conditions: u(0, t) = 0, u(L, t) = 0
    x_train_BC_lb = np.hstack((X[:, 0:1], T[:, 0:1]))
    u_exact_BC_lb = u_exact[:, 0:1]
    x_train_BC_ub = np.hstack((X[:, -1:], T[:, -1:]))
    u_exact_BC_ub = u_exact[:, -1:]

    x_train_IC_BC = np.vstack([x_train_IC, x_train_BC_lb, x_train_BC_ub])
    y_train_IC_BC = np.vstack([u_exact_IC, u_exact_BC_lb, u_exact_BC_ub])

    # * Sample collocation points via LHS
    n_collocation = 2000
    x_train = lower_boundary + (upper_boundary - lower_boundary) * lhs(2, n_collocation)
    x_train = np.vstack((x_train, x_train_IC_BC))

    # * Subsample IC/BC
    n_bc_points = 150
    idx = np.random.choice(x_train_IC_BC.shape[0], n_bc_points, replace=False)
    x_train_IC_BC = x_train_IC_BC[idx, :]
    y_train_IC_BC = y_train_IC_BC[idx, :]

    # * Save
    np.save(os.path.join(path, "x_train_IC_BC"), x_train_IC_BC)
    np.save(os.path.join(path, "y_train_IC_BC"), y_train_IC_BC)
    np.save(os.path.join(path, "x_train"), x_train)
    np.save(os.path.join(path, "u_exact"), u_exact)
    np.save(os.path.join(path, "x_star"), x_star)


if __name__ == "__main__":
    generate_dataset()
