""" This work is unfinished, yet.
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from pyDOE import lhs


PLATE_LENGTH = 50
MAX_ITER_TIME = 50
ALPHA = 2


def generate_dataset(
    x_domain_lower_boundary: float,
    x_domain_upper_boundary: float,
    x_domain_resolution: int,
    y_domain_lower_boundary: float,
    y_domain_upper_boundary: float,
    y_domain_resolution: int,
    t_domain_lower_boundary: float,
    t_domain_upper_boundary: float,
    t_domain_resolution: int,
    sanity_check: bool = False,
    priority_of_IC: bool = False,
    priority_of_left_or_right: bool = False,
    path: str = "src/heat_eq_2d/data",
) -> None:
    if not priority_of_IC:
        IC_exclusion = 0
    else:
        IC_exclusion = 1

    if not priority_of_left_or_right:
        x_exclusion_start = 0
        x_exclusion_end = None
        y_exclusion_start = 1
        y_exclusion_end = -1
    else:
        x_exclusion_start = 1
        x_exclusion_end = -1
        y_exclusion_start = 0
        y_exclusion_end = None

    # FDM solution
    y_train = fdm_solution(x_domain_upper_boundary, t_domain_upper_boundary, ALPHA)

    # Preprocessing
    x_domain = np.linspace(
        x_domain_lower_boundary, x_domain_upper_boundary, x_domain_resolution
    )
    y_domain = np.linspace(
        y_domain_lower_boundary, y_domain_upper_boundary, y_domain_resolution
    )
    t_domain = np.linspace(
        t_domain_lower_boundary, t_domain_upper_boundary, t_domain_resolution
    )

    X, Y, T = np.meshgrid(x_domain, y_domain, t_domain, indexing="ij")

    # (x_domain_resolution * y_domain_resolution * t_domain_resolution, 3)
    x_star = np.hstack((
        X.reshape(-1, 1), Y.reshape(-1, 1), T.reshape(-1, 1)
    ))

    #assert y_train.shape[0] == x_star.shape[0]

    # Initial condition
    x_train_IC = np.hstack((
        X[:, :, 0].reshape(-1, 1),
        Y[:, :, 0].reshape(-1, 1),
        T[:, :, 0].reshape(-1, 1),
    ))
    # y is the shape of x_domain_resolution * y_domain_resolution
    assert x_train_IC.shape[0] == x_domain_resolution * y_domain_resolution
    y_train_IC = np.zeros(x_train_IC.shape[0]).reshape(-1, 1)

    # Boundary condition
    # upper =   (x, y=-1, t)
    # lower =   (x, y=0, t)
    # left  =   (x=0, y, t)
    # right =   (x=-1, y, t)
    x_train_BC_left_boundary = np.hstack((
        X[0, y_exclusion_start:y_exclusion_end, IC_exclusion:].reshape(-1, 1),
        Y[0, y_exclusion_start:y_exclusion_end, IC_exclusion:].reshape(-1, 1),
        T[0, y_exclusion_start:y_exclusion_end, IC_exclusion:].reshape(-1, 1),
    ))
    x_train_BC_right_boundary = np.hstack((
        X[-1, y_exclusion_start:y_exclusion_end, IC_exclusion:].reshape(-1, 1),
        Y[-1, y_exclusion_start:y_exclusion_end, IC_exclusion:].reshape(-1, 1),
        T[-1, y_exclusion_start:y_exclusion_end, IC_exclusion:].reshape(-1, 1),
    ))
    x_train_BC_lower_boundary = np.hstack((
        X[x_exclusion_start:x_exclusion_end, 0, IC_exclusion:].reshape(-1, 1),
        Y[x_exclusion_start:x_exclusion_end, 0, IC_exclusion:].reshape(-1, 1),
        T[x_exclusion_start:x_exclusion_end, 0, IC_exclusion:].reshape(-1, 1),
    ))
    x_train_BC_upper_boundary = np.hstack((
        X[x_exclusion_start:x_exclusion_end, -1, IC_exclusion:].reshape(-1, 1),
        Y[x_exclusion_start:x_exclusion_end, -1, IC_exclusion:].reshape(-1, 1),
        T[x_exclusion_start:x_exclusion_end, -1, IC_exclusion:].reshape(-1, 1),
    ))
    expected_lr = (
        (t_domain_resolution - IC_exclusion)
        * (y_domain_resolution - 2 * y_exclusion_start)
    )
    assert x_train_BC_left_boundary.shape[0] == expected_lr
    assert x_train_BC_right_boundary.shape[0] == expected_lr
    expected_tb = (
        (t_domain_resolution - IC_exclusion)
        * (x_domain_resolution - 2 * x_exclusion_start)
    )
    assert x_train_BC_lower_boundary.shape[0] == expected_tb
    assert x_train_BC_upper_boundary.shape[0] == expected_tb

    # This will be a parameter
    y_train_BC_left_boundary = (
        np.ones(x_train_BC_left_boundary.shape[0]).reshape(-1, 1) * 100
    )
    y_train_BC_right_boundary = (
        np.ones(x_train_BC_right_boundary.shape[0]).reshape(-1, 1) * 0
    )
    y_train_BC_lower_boundary = (
        np.ones(x_train_BC_lower_boundary.shape[0]).reshape(-1, 1) * 0
    )
    y_train_BC_upper_boundary = (
        np.ones(x_train_BC_upper_boundary.shape[0]).reshape(-1, 1) * 100
    )

    # Final assembly, BC incorporates IC
    all_x_train_IC_BC = np.vstack([
        x_train_IC,
        x_train_BC_left_boundary,
        x_train_BC_right_boundary,
        x_train_BC_lower_boundary,
        x_train_BC_upper_boundary,
    ])
    all_y_train_IC_BC = np.vstack([
        y_train_IC,
        y_train_BC_left_boundary,
        y_train_BC_right_boundary,
        y_train_BC_lower_boundary,
        y_train_BC_upper_boundary,
    ])

    lower_boundary = x_star.min(axis=0)
    upper_boundary = x_star.max(axis=0)

    x_train = lower_boundary + (upper_boundary - lower_boundary) * lhs(
        x_star.min(axis=0).shape[0],
        12500,
    )
    x_train = np.vstack((x_train, all_x_train_IC_BC))

    # * Final data
    n_bc_points = 5000
    idx = np.random.choice(all_x_train_IC_BC.shape[0], n_bc_points, replace=False)
    x_train_IC_BC = all_x_train_IC_BC[idx, :]
    y_train_IC_BC = all_y_train_IC_BC[idx, :]

    np.save(os.path.join(path, "x_train_IC_BC"), x_train_IC_BC)
    np.save(os.path.join(path, "y_train_IC_BC"), y_train_IC_BC)
    np.save(os.path.join(path, "x_train"), x_train)
    np.save(os.path.join(path, "x_star"), x_star)


def plotheatmap(u_k, k, delta_t):
    # Clear the current plot figure
    plt.clf()

    plt.title(f"Temperature at t = {k * delta_t:.3f} unit time")
    plt.xlabel("x")
    plt.ylabel("y")

    # This is to plot u_k (u at time-step k)
    plt.pcolormesh(u_k, cmap="viridis", vmin=0, vmax=100)
    plt.colorbar()

    return plt


def fdm_solution(
    plate_length: int,
    max_iter_time: int,
    alpha: float,
    path: str = "src/heat_eq_2d/data",
) -> np.ndarray:
    """
    Reference:
    https://levelup.gitconnected.com/solving-2d-heat-equation-numerically-using-python-3334004aa01a
    (Universitas Padjadjaran)
    """
    delta_x = 1

    delta_t = (delta_x ** 2)/(4 * alpha)
    gamma = (alpha * delta_t) / (delta_x ** 2)

    # Initialize solution: the grid of u(k, i, j)
    u = np.empty((max_iter_time, plate_length, plate_length))

    # Initial condition everywhere inside the grid
    u_initial = 0
    u.fill(u_initial)

    # Boundary conditions
    # Convention: u[k, i, j] where i indexes x-axis, j indexes y-axis
    # Must match PINN BCs in generate_dataset():
    # left(x=0)=100, right(x=max)=0, lower(y=0)=0, upper(y=max)=100
    u_left = 100.0    # x = 0 (i = 0)
    u_right = 0.0     # x = max (i = plate_length - 1)
    u_bottom = 0.0    # y = 0 (j = 0)
    u_top = 100.0      # y = max (j = plate_length - 1)

    # Set the boundary conditions
    u[:, :1, :] = u_left                       # i = 0 (x = 0)
    u[:, (plate_length - 1):, :] = u_right     # i = max (x = max)
    u[:, :, :1] = u_bottom                     # j = 0 (y = 0)
    u[:, :, (plate_length - 1):] = u_top       # j = max (y = max)

    # todo: clean-up
    def calculate(u):
        for k in range(0, max_iter_time - 1, 1):
            for i in range(1, plate_length - 1, delta_x):
                for j in range(1, plate_length - 1, delta_x):
                    u[k + 1, i, j] = gamma * (
                        u[k][i + 1][j]
                        + u[k][i - 1][j]
                        + u[k][i][j + 1]
                        + u[k][i][j - 1]
                        - 4 * u[k][i][j]
                    ) + u[k][i][j]

        return u

    y_train = calculate(u)

    np.save(os.path.join(path, "y_train"), y_train)

    # Visual sanity check
    #def animate(k):
    #    plotheatmap(u[k], k, delta_t)

    #anim = animation.FuncAnimation(
    #    plt.figure(),
    #    animate,
    #    interval=1,
    #    frames=max_iter_time,
    #    repeat=False,
    #)
    #anim.save("src/heat_eq_2d/heat_equation_solution.gif")

    return y_train


if __name__ == "__main__":
    generate_dataset(
        x_domain_lower_boundary=0,
        x_domain_upper_boundary=PLATE_LENGTH,
        x_domain_resolution=50,
        y_domain_lower_boundary=0,
        y_domain_upper_boundary=PLATE_LENGTH,
        y_domain_resolution=50,
        t_domain_lower_boundary=0,
        t_domain_upper_boundary=MAX_ITER_TIME,
        t_domain_resolution=50,
    )

    fdm_solution(
        plate_length=PLATE_LENGTH,
        max_iter_time=MAX_ITER_TIME,
        alpha=ALPHA,
    )
