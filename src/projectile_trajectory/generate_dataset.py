""" Generate dataset for projectile trajectory with optional linear drag.

Governing ODEs:
    No drag:     x_tt = 0,            y_tt = -g
    Linear drag: x_tt = -beta * x_t,  y_tt = -g - beta * y_t

where (x, y) is position, g is gravitational acceleration, and
beta = b/m is the drag coefficient divided by mass.

Initial conditions:
    x(0) = 0,  y(0) = 0
    x_t(0) = v0 * cos(alpha),  y_t(0) = v0 * sin(alpha)

Cases:
    1. no_drag:     Ideal ballistic trajectory (beta = 0)
    2. linear_drag: Linear drag proportional to velocity (beta > 0)

References:
    Meriam, J.L. & Kraige, L.G. (2012).
        "Engineering Mechanics: Dynamics." Wiley, 7th edition.
"""
import os

import numpy as np
from pyDOE import lhs


def exact_solution(t, case, v0=10.0, alpha=np.pi / 4, g=9.81, beta=0.5):
    """Compute exact position, velocity, and acceleration for a given case.

    Args:
        t: Time array, shape (n,) or (n, 1).
        case: One of "no_drag", "linear_drag".
        v0: Initial speed.
        alpha: Launch angle in radians.
        g: Gravitational acceleration.
        beta: Linear drag coefficient (b/m). Used only for linear_drag case.

    Returns:
        (x, y, vx, vy, ax, ay): Position, velocity, and acceleration arrays.
    """
    t = np.asarray(t).flatten()
    vx0 = v0 * np.cos(alpha)
    vy0 = v0 * np.sin(alpha)

    if case == "no_drag":
        x = vx0 * t
        y = vy0 * t - 0.5 * g * t**2
        vx = vx0 * np.ones_like(t)
        vy = vy0 - g * t
        ax = np.zeros_like(t)
        ay = -g * np.ones_like(t)

    elif case == "linear_drag":
        exp_bt = np.exp(-beta * t)
        x = (vx0 / beta) * (1.0 - exp_bt)
        y = (1.0 / beta) * (vy0 + g / beta) * (1.0 - exp_bt) - (g / beta) * t
        vx = vx0 * exp_bt
        vy = (vy0 + g / beta) * exp_bt - g / beta
        ax = -beta * vx
        ay = -g - beta * vy

    else:
        raise ValueError(
            f"Unknown case: '{case}'. Choose from: no_drag, linear_drag"
        )

    return x, y, vx, vy, ax, ay


def _flight_time(case, v0=10.0, alpha=np.pi / 4, g=9.81, beta=0.5):
    """Compute approximate flight time (when y returns to zero).

    Args:
        case: One of "no_drag", "linear_drag".
        v0, alpha, g, beta: Physical parameters.

    Returns:
        T_flight: Approximate flight time.
    """
    if case == "no_drag":
        return 2.0 * v0 * np.sin(alpha) / g
    else:
        # Numerical root finding for y(T) = 0
        from scipy.optimize import brentq
        vy0 = v0 * np.sin(alpha)

        def y_func(t):
            if t == 0:
                return 0.0
            exp_bt = np.exp(-beta * t)
            return (1.0 / beta) * (vy0 + g / beta) * (1.0 - exp_bt) - (g / beta) * t

        # Find upper bracket: y goes negative eventually
        t_upper = 2.0 * v0 * np.sin(alpha) / g  # no-drag flight time as starting guess
        while y_func(t_upper) > 0:
            t_upper *= 1.5
        return brentq(y_func, 0.01, t_upper)


def _build_ic_arrays(case, v0=10.0, alpha=np.pi / 4):
    """Build initial condition arrays with order and output-index indicators.

    Returns:
        x_train_IC: shape (n_ic, 3) -- col 0: time, col 1: derivative order, col 2: output index
        y_train_IC: shape (n_ic, 1) -- target value
    """
    vx0 = v0 * np.cos(alpha)
    vy0 = v0 * np.sin(alpha)

    # Each row: [t, derivative_order, output_index]
    # derivative_order: 0 = position, 1 = velocity
    # output_index: 0 = x, 1 = y
    ic_x = np.array([
        [0.0, 0, 0],  # x(0) = 0
        [0.0, 0, 1],  # y(0) = 0
        [0.0, 1, 0],  # vx(0) = v0 * cos(alpha)
        [0.0, 1, 1],  # vy(0) = v0 * sin(alpha)
    ])
    ic_y = np.array([
        [0.0],
        [0.0],
        [vx0],
        [vy0],
    ])

    return ic_x, ic_y


def generate_dataset(
    case="no_drag",
    path="src/projectile_trajectory/data",
    v0=10.0,
    alpha=np.pi / 4,
    g=9.81,
    beta=0.5,
):
    """Generate and save training/evaluation data for the projectile trajectory.

    Args:
        case: One of "no_drag", "linear_drag".
        path: Directory to save data files.
        v0: Initial speed.
        alpha: Launch angle (radians).
        g: Gravitational acceleration.
        beta: Linear drag coefficient (b/m).
    """
    T_flight = _flight_time(case, v0=v0, alpha=alpha, g=g, beta=beta)

    # Fine evaluation grid
    t_star = np.linspace(0, T_flight, 201).reshape(-1, 1)
    x_exact, y_exact, vx_exact, vy_exact, ax_exact, ay_exact = exact_solution(
        t_star, case, v0=v0, alpha=alpha, g=g, beta=beta,
    )

    # Collocation points via Latin Hypercube Sampling
    n_collocation = 300
    t_train = lhs(1, n_collocation) * T_flight

    # Initial conditions with order and output-index indicator
    ic_x, ic_y = _build_ic_arrays(case, v0=v0, alpha=alpha)

    # Replicate IC points for stable batching (only 4 unique ICs)
    n_replicate = 50
    x_train_IC = np.tile(ic_x, (n_replicate, 1))
    y_train_IC = np.tile(ic_y, (n_replicate, 1))

    # Non-dimensionalization
    # Scales: T_ref = T_flight, V_ref = v0, L_ref = v0 * T_flight
    T_ref = T_flight
    V_ref = v0
    L_ref = v0 * T_flight

    # Scale time inputs to [0, 1]
    t_train /= T_ref
    t_star /= T_ref

    # Scale IC velocity targets (position ICs are 0, no change needed)
    velocity_mask = x_train_IC[:, 1] == 1  # derivative_order == 1
    y_train_IC[velocity_mask] /= V_ref

    # Save
    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, "t_train"), t_train)
    np.save(os.path.join(path, "x_train_IC"), x_train_IC)
    np.save(os.path.join(path, "y_train_IC"), y_train_IC)
    np.save(os.path.join(path, "t_star"), t_star)
    np.save(os.path.join(path, "x_exact"), x_exact)
    np.save(os.path.join(path, "y_exact"), y_exact)
    np.save(os.path.join(path, "vx_exact"), vx_exact)
    np.save(os.path.join(path, "vy_exact"), vy_exact)
    np.save(os.path.join(path, "scaling"), {
        "T_ref": T_ref, "V_ref": V_ref, "L_ref": L_ref,
    })


if __name__ == "__main__":
    for case in ["no_drag", "linear_drag"]:
        generate_dataset(
            case=case,
            path=f"src/projectile_trajectory/data/{case}",
        )
