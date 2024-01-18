"""
Reference:
https://aquaulb.github.io/book_solving_pde_mooc/solving_pde_mooc/notebooks/04_PartialDifferentialEquations/04_03_Diffusion_Explicit.html (Free University of Brussels)
"""
import os

from typing import List

import numpy as np
import plotly.express as px

from pyDOE import lhs


def exact_solution(
    x: List[float],
    t: float,
    alpha: float,
):
    """ Returns the exact solution of the 1D heat equation with heat source term sin(np.pi*x) and initial condition sin(2*np.pi*x).

    Args:
        x (List[float]): xaxis
        t (float): time
        alpha (float): coeff

    Returns:
        temperature
    """
    return (
        np.exp(-4 * np.pi ** 2 * alpha * t) * np.sin(2 * np.pi * x) 
        + 2.0 * (1 - np.exp(-np.pi ** 2 * alpha * t)) * np.sin(np.pi * x) / (np.pi ** 2 * alpha)
    )


def generate_dataset(path: str = "src/continuous_time/heat_eq_1d/data",):
    # Physical parameters
    alpha = 0.1                             # Heat transfer coefficient
    lx = 1.                                 # Size of computational domain

    # Grid parameters
    nx = 21                                 # number of grid points 
    x_domain = np.linspace(0., lx, nx)      # coordinates of grid points

    u_list = []
    time_domain = np.linspace(0, 5, 100)
    for time in time_domain:
        u = exact_solution(x=x_domain, t=time, alpha=alpha)
        u_list.append(u)

    y_train = np.vstack(u_list)

    fig = px.imshow(y_train, origin="lower")
    fig.write_html("src/continuous_time/heat_eq_1d/plots/analytical_solution.html")

    # * Initial and boundary conditions
    X, T = np.meshgrid(x_domain, time_domain)  # (100, 21), (100, 21)

    X_star = np.hstack((
        X.flatten().reshape(-1, 1),
        T.flatten().reshape(-1, 1),
    ))

    lower_boundary = X_star.min(axis=0)
    upper_boundary = X_star.max(axis=0)

    # initial conditions
    x_train_IC = np.hstack((X[0:1, :].T, T[0:1, :].T))
    y_train_IC = y_train[0, :].reshape(-1, 1)

    # boundary conditions
    x_train_BC_lb = np.hstack((X[:, 0:1], T[:, 0:1]))
    y_train_BC_lb = y_train[:, 0:1]
    x_train_BC_ub = np.hstack((X[:, -1:], T[:, -1:]))
    y_train_BC_ub = y_train[:, -1:]

    x_train_BC = np.vstack([x_train_IC, x_train_BC_lb, x_train_BC_ub])
    y_train_BC = np.vstack([y_train_IC, y_train_BC_lb, y_train_BC_ub])

    # * Sample collocation
    x_train = lower_boundary + (upper_boundary - lower_boundary) * lhs(2, 1000)
    x_train = np.vstack((x_train, x_train_BC))  # (10000+456=10456, 2)


    # * Final data
    idx = np.random.choice(x_train_BC.shape[0], 100, replace=False)
    x_train_BC = x_train_BC[idx, :]   # (100, 2)
    y_train_BC = y_train_BC[idx, :]       # (100, 1)

    # * Save
    np.save(os.path.join(path, "x_train_BC"), x_train_BC)
    np.save(os.path.join(path, "y_train_BC"), y_train_BC)
    np.save(os.path.join(path, "x_train"), x_train)
    np.save(os.path.join(path, "y_train"), y_train)
    np.save(os.path.join(path, "X_star"), X_star)


if __name__ == "__main__":
    generate_dataset()
