""" Generate dataset for Euler-Bernoulli beam deflection.

Governing ODE (4th order):
    EI * d4w/dx4 = q(x)

where w(x) is transverse deflection, E is Young's modulus,
I is second moment of area, and q(x) is distributed load.

Derived quantities:
    slope:   theta(x) = dw/dx
    moment:  M(x)     = EI * d2w/dx2
    shear:   V(x)     = EI * d3w/dx3

Load cases:
    1. cantilever_point_load: Cantilever beam with point load P at free end
    2. simply_supported_udl:  Simply supported beam with uniform distributed load q0
    3. cantilever_udl:        Cantilever beam with uniform distributed load q0

References:
    Timoshenko, S.P. & Gere, J.M. (1961).
        "Theory of Elastic Stability." McGraw-Hill.

    Beer, F.P., Johnston, E.R. & DeWolf, J.T. (2012).
        "Mechanics of Materials." McGraw-Hill, 7th edition.

    Euler-Bernoulli beam theory.
        https://en.wikipedia.org/wiki/Euler%E2%80%93Bernoulli_beam_theory
"""
import os

import numpy as np
from pyDOE import lhs


def exact_solution(x, load_case, L=1.0, EI=1.0, P=1.0, q0=1.0):
    """Compute exact deflection, slope, moment, and shear for a given load case.

    Args:
        x: Spatial coordinates along beam, shape (n,) or (n, 1).
        load_case: One of "cantilever_point_load", "simply_supported_udl",
            "cantilever_udl".
        L: Beam length.
        EI: Flexural rigidity (Young's modulus * second moment of area).
        P: Point load magnitude (used for cantilever_point_load).
        q0: Distributed load intensity (used for udl cases).

    Returns:
        (w, theta, M, V): Deflection, slope, moment, and shear arrays.
    """
    x = np.asarray(x).flatten()

    if load_case == "cantilever_point_load":
        # Fixed at x=0, point load P downward at x=L
        # w(x)     = P/(6EI) * (3L x^2 - x^3)
        # theta(x) = P/(6EI) * (6L x - 3x^2)
        # M(x)     = P * (L - x)
        # V(x)     = -P  (constant)
        w = (P / (6 * EI)) * (3 * L * x**2 - x**3)
        theta = (P / (6 * EI)) * (6 * L * x - 3 * x**2)
        M = P * (L - x)
        V = -P * np.ones_like(x)

    elif load_case == "simply_supported_udl":
        # Pinned at x=0 and x=L, uniform load q0
        # w(x)     = q0/(24EI) * (x^4 - 2L x^3 + L^3 x)
        # theta(x) = q0/(24EI) * (4x^3 - 6L x^2 + L^3)
        # M(x)     = q0/2 * (L x - x^2)
        # V(x)     = q0 * (L/2 - x)
        w = (q0 / (24 * EI)) * (x**4 - 2 * L * x**3 + L**3 * x)
        theta = (q0 / (24 * EI)) * (4 * x**3 - 6 * L * x**2 + L**3)
        M = (q0 / 2) * (L * x - x**2)
        V = q0 * (L / 2 - x)

    elif load_case == "cantilever_udl":
        # Fixed at x=0, free at x=L, uniform load q0
        # w(x)     = q0/(24EI) * (x^4 - 4L x^3 + 6L^2 x^2)
        # theta(x) = q0/(24EI) * (4x^3 - 12L x^2 + 12L^2 x)
        # M(x)     = q0/2 * (L - x)^2
        # V(x)     = -q0 * (L - x)
        w = (q0 / (24 * EI)) * (x**4 - 4 * L * x**3 + 6 * L**2 * x**2)
        theta = (q0 / (24 * EI)) * (4 * x**3 - 12 * L * x**2 + 12 * L**2 * x)
        M = (q0 / 2) * (L - x) ** 2
        V = -q0 * (L - x)

    else:
        raise ValueError(
            f"Unknown load_case: '{load_case}'. "
            "Choose from: cantilever_point_load, simply_supported_udl, cantilever_udl"
        )

    return w, theta, M, V


def _build_bc_arrays(load_case, L=1.0, EI=1.0, P=1.0, q0=1.0):
    """Build boundary condition arrays with order indicator column.

    Returns:
        x_train_BC: shape (n_bc, 2) -- col 0: x position, col 1: derivative order
        y_train_BC: shape (n_bc, 1) -- target value
    """
    # Each row: [x_position, derivative_order], [target_value]
    # order 0: w, order 1: dw/dx, order 2: d2w/dx2, order 3: d3w/dx3

    if load_case == "cantilever_point_load":
        bc_x = np.array([
            [0.0, 0],  # w(0) = 0
            [0.0, 1],  # dw/dx(0) = 0
            [L, 2],    # d2w/dx2(L) = 0  (M=0 at free end)
            [L, 3],    # d3w/dx3(L) = -P/EI  (V=-P at free end)
        ])
        bc_y = np.array([
            [0.0],
            [0.0],
            [0.0],
            [-P / EI],
        ])

    elif load_case == "simply_supported_udl":
        bc_x = np.array([
            [0.0, 0],  # w(0) = 0
            [0.0, 2],  # d2w/dx2(0) = 0  (M=0 at pin)
            [L, 0],    # w(L) = 0
            [L, 2],    # d2w/dx2(L) = 0  (M=0 at pin)
        ])
        bc_y = np.array([
            [0.0],
            [0.0],
            [0.0],
            [0.0],
        ])

    elif load_case == "cantilever_udl":
        bc_x = np.array([
            [0.0, 0],  # w(0) = 0
            [0.0, 1],  # dw/dx(0) = 0
            [L, 2],    # d2w/dx2(L) = 0  (M=0 at free end)
            [L, 3],    # d3w/dx3(L) = 0  (V=0 at free end)
        ])
        bc_y = np.array([
            [0.0],
            [0.0],
            [0.0],
            [0.0],
        ])

    else:
        raise ValueError(f"Unknown load_case: '{load_case}'.")

    return bc_x, bc_y


def generate_dataset(
    load_case="cantilever_point_load",
    path="src/euler_bernoulli_beam/data",
):
    L = 1.0
    EI = 1.0
    P = 1.0
    q0 = 1.0

    # Fine evaluation grid
    x_star = np.linspace(0, L, 201).reshape(-1, 1)
    w_exact, theta_exact, M_exact, V_exact = exact_solution(
        x_star, load_case, L=L, EI=EI, P=P, q0=q0,
    )

    # Collocation points via Latin Hypercube Sampling
    n_collocation = 500
    x_train = lhs(1, n_collocation) * L

    # Boundary conditions with order indicator
    bc_x, bc_y = _build_bc_arrays(load_case, L=L, EI=EI, P=P, q0=q0)

    # Replicate BC points for stable batching (only 4 unique BCs)
    n_replicate = 50
    x_train_BC = np.tile(bc_x, (n_replicate, 1))
    y_train_BC = np.tile(bc_y, (n_replicate, 1))

    # Save
    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, "x_train"), x_train)
    np.save(os.path.join(path, "x_train_BC"), x_train_BC)
    np.save(os.path.join(path, "y_train_BC"), y_train_BC)
    np.save(os.path.join(path, "x_star"), x_star)
    np.save(os.path.join(path, "w_exact"), w_exact)
    np.save(os.path.join(path, "theta_exact"), theta_exact)
    np.save(os.path.join(path, "M_exact"), M_exact)
    np.save(os.path.join(path, "V_exact"), V_exact)


if __name__ == "__main__":
    generate_dataset()
