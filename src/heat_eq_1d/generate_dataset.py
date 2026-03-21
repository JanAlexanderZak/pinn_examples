"""
Reference:
https://aquaulb.github.io/book_solving_pde_mooc/solving_pde_mooc/notebooks/
04_PartialDifferentialEquations/04_03_Diffusion_Explicit.html
(Free University of Brussels)
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
    alpha: float,
):
    """ Returns the exact solution of the 1D heat equation
    with heat source term sin(np.pi*x)
    and initial condition sin(2*np.pi*x).

    Args:
        x (List[float]): xaxis
        t (float): time
        alpha (float): coeff

    Returns:
        temperature
    """
    return (
        np.exp(-4 * np.pi ** 2 * alpha * t) * np.sin(2 * np.pi * x)
        + 2.0
        * (1 - np.exp(-np.pi ** 2 * alpha * t))
        * np.sin(np.pi * x)
        / (np.pi ** 2 * alpha)
    )


def generate_dataset(path: str = "src/heat_eq_1d/data",):
    # Physical parameters
    alpha = 0.1                                     # Heat transfer coefficient
    domain_length = 1.                              # Size of computational domain

    # Grid parameters
    x_resolution = 21                               # number of grid points
    x_domain = np.linspace(0., domain_length, x_resolution)

    time_domain = np.linspace(0, 5, 100)
    u_exact = np.vstack([
        exact_solution(x=x_domain, t=time, alpha=alpha)
        for time in time_domain
    ])

    with pinn_style():
        fig, ax = plt.subplots()
        im = ax.imshow(u_exact, origin="lower", cmap=CMAP_SEQUENTIAL, aspect="auto")
        ax.set_xlabel("t")
        ax.set_ylabel("x")
        fig.colorbar(im, ax=ax)
        save_figure(fig, "src/heat_eq_1d/plots/analytical_solution.png")

    # * Initial and boundary conditions
    X, T = np.meshgrid(x_domain, time_domain)  # (100, 21), (100, 21)

    x_star = np.hstack((
        X.flatten().reshape(-1, 1),
        T.flatten().reshape(-1, 1),
    ))

    lower_boundary = x_star.min(axis=0)
    upper_boundary = x_star.max(axis=0)

    # initial conditions
    x_train_IC = np.hstack((X[0:1, :].T, T[0:1, :].T))
    u_exact_IC = u_exact[0, :].reshape(-1, 1)

    # boundary conditions
    x_train_BC_lb = np.hstack((X[:, 0:1], T[:, 0:1]))
    u_exact_BC_lb = u_exact[:, 0:1]
    x_train_BC_ub = np.hstack((X[:, -1:], T[:, -1:]))
    u_exact_BC_ub = u_exact[:, -1:]

    x_train_IC_BC = np.vstack([x_train_IC, x_train_BC_lb, x_train_BC_ub])
    y_train_IC_BC = np.vstack([u_exact_IC, u_exact_BC_lb, u_exact_BC_ub])

    # * Sample collocation
    x_train = lower_boundary + (upper_boundary - lower_boundary) * lhs(2, 1000)
    x_train = np.vstack((x_train, x_train_IC_BC))

    # * Final data
    n_bc_points = 100
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
