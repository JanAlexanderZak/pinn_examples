""" Generate dataset for Kovasznay flow (steady 2D incompressible Navier-Stokes).

Kovasznay (1948) exact solution for steady incompressible N-S at low Reynolds number.
Domain: x in [-0.5, 1.0], y in [-0.5, 1.5]

Exact solution:
    lambda = 1/(2*nu) - sqrt(1/(4*nu^2) + 4*pi^2)
    u(x,y) = 1 - exp(lambda*x) * cos(2*pi*y)
    v(x,y) = lambda/(2*pi) * exp(lambda*x) * sin(2*pi*y)
    p(x,y) = 0.5 * (1 - exp(2*lambda*x))

Reference: Kovasznay, L.I.G. (1948), "Laminar flow behind a two-dimensional grid"
"""
import os

import numpy as np

from pyDOE import lhs


def exact_solution(x, y, nu):
    lam = 1.0 / (2.0 * nu) - np.sqrt(1.0 / (4.0 * nu**2) + 4.0 * np.pi**2)

    u = 1.0 - np.exp(lam * x) * np.cos(2.0 * np.pi * y)
    v = lam / (2.0 * np.pi) * np.exp(lam * x) * np.sin(2.0 * np.pi * y)
    p = 0.5 * (1.0 - np.exp(2.0 * lam * x))

    return u, v, p


def generate_dataset(path: str = "src/navier_stokes_kovasznay/data"):
    # Physical parameters
    Re = 20.0
    nu = 1.0 / Re

    # Domain
    x_min, x_max = -0.5, 1.0
    y_min, y_max = -0.5, 1.5

    # Grid for exact solution and prediction
    x_resolution = 80
    y_resolution = 80
    x_domain = np.linspace(x_min, x_max, x_resolution)
    y_domain = np.linspace(y_min, y_max, y_resolution)
    X, Y = np.meshgrid(x_domain, y_domain)

    u_exact, v_exact, p_exact = exact_solution(X, Y, nu)

    x_star = np.hstack((
        X.flatten().reshape(-1, 1),
        Y.flatten().reshape(-1, 1),
    ))

    lower_boundary = np.array([x_min, y_min])
    upper_boundary = np.array([x_max, y_max])

    # * Boundary conditions (all four edges)
    # Bottom edge: y = y_min
    x_bc_bottom = np.hstack((
        x_domain.reshape(-1, 1), np.full((x_resolution, 1), y_min)
    ))
    u_bc_bottom, v_bc_bottom, p_bc_bottom = exact_solution(
        x_bc_bottom[:, 0], x_bc_bottom[:, 1], nu
    )

    # Top edge: y = y_max
    x_bc_top = np.hstack((x_domain.reshape(-1, 1), np.full((x_resolution, 1), y_max)))
    u_bc_top, v_bc_top, p_bc_top = exact_solution(x_bc_top[:, 0], x_bc_top[:, 1], nu)

    # Left edge: x = x_min
    x_bc_left = np.hstack((np.full((y_resolution, 1), x_min), y_domain.reshape(-1, 1)))
    u_bc_left, v_bc_left, p_bc_left = exact_solution(
        x_bc_left[:, 0], x_bc_left[:, 1], nu
    )

    # Right edge: x = x_max
    x_bc_right = np.hstack((np.full((y_resolution, 1), x_max), y_domain.reshape(-1, 1)))
    u_bc_right, v_bc_right, p_bc_right = exact_solution(
        x_bc_right[:, 0], x_bc_right[:, 1], nu
    )

    x_train_BC = np.vstack([x_bc_bottom, x_bc_top, x_bc_left, x_bc_right])
    y_train_BC = np.vstack([
        np.column_stack([u_bc_bottom, v_bc_bottom, p_bc_bottom]),
        np.column_stack([u_bc_top, v_bc_top, p_bc_top]),
        np.column_stack([u_bc_left, v_bc_left, p_bc_left]),
        np.column_stack([u_bc_right, v_bc_right, p_bc_right]),
    ])

    # * Sample collocation points via LHS
    n_collocation = 5000
    x_train = lower_boundary + (upper_boundary - lower_boundary) * lhs(2, n_collocation)
    x_train = np.vstack((x_train, x_train_BC))

    # * Subsample BC
    n_bc_points = 200
    idx = np.random.choice(x_train_BC.shape[0], n_bc_points, replace=False)
    x_train_BC = x_train_BC[idx, :]
    y_train_BC = y_train_BC[idx, :]

    # * Save
    np.save(os.path.join(path, "x_train_BC"), x_train_BC)
    np.save(os.path.join(path, "y_train_BC"), y_train_BC)
    np.save(os.path.join(path, "x_train"), x_train)
    np.save(os.path.join(path, "x_star"), x_star)
    np.save(os.path.join(path, "u_exact"), u_exact)
    np.save(os.path.join(path, "v_exact"), v_exact)
    np.save(os.path.join(path, "p_exact"), p_exact)


if __name__ == "__main__":
    generate_dataset()
