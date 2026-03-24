""" Visualization for projectile trajectory PINN results.

Produces comparison plots of trajectory, position, velocity,
and speed (exact vs. PINN prediction), plus an absolute error plot.
"""
import numpy as np
import torch

from src.visualization import plot_line, pinn_style, save_figure, FIGSIZE_LINE_PLOT, COLOR_EXACT, COLOR_PINN, COLOR_CYCLE
from src.projectile_trajectory.generate_dataset import exact_solution, _flight_time

import matplotlib.pyplot as plt


def compute_trajectory_derivatives(model, t_np):
    """Compute position, velocity from the trained model using autograd.

    The model operates in non-dimensional space. This function handles
    scaling the input and unscaling the outputs to return physical values.

    Args:
        model: Trained ProjectilePINNRegressor.
        t_np: Numpy array of physical time values, shape (n,) or (n, 1).

    Returns:
        (x, y, vx, vy): Numpy arrays in physical units.
    """
    T_ref = model.hparams.T_ref
    V_ref = model.hparams.V_ref
    L_ref = model.hparams.L_ref

    t_np = np.asarray(t_np).flatten()
    # Scale time to non-dimensional input
    t_nd = t_np / T_ref
    t_t = torch.tensor(t_nd.reshape(-1, 1), dtype=torch.float32, requires_grad=True)

    xy = model.net(t_t)
    ones = torch.ones(t_t.shape[0], 1)

    x_t = torch.autograd.grad(xy[:, 0:1], t_t, ones, retain_graph=True, create_graph=True)[0]
    y_t = torch.autograd.grad(xy[:, 1:2], t_t, ones, retain_graph=True, create_graph=True)[0]

    # Unscale: positions by L_ref, velocities by V_ref (= L_ref / T_ref)
    x_np = xy[:, 0].detach().numpy() * L_ref
    y_np = xy[:, 1].detach().numpy() * L_ref
    vx_np = x_t.detach().numpy().flatten() * V_ref
    vy_np = y_t.detach().numpy().flatten() * V_ref

    return x_np, y_np, vx_np, vy_np


def visualize(
    model,
    case="no_drag",
    save_dir="src/figures",
    v0=10.0,
    alpha=np.pi / 4,
    g=9.81,
    beta=0.5,
):
    """Generate all projectile visualization plots.

    Args:
        model: Trained ProjectilePINNRegressor.
        case: Case string ("no_drag" or "linear_drag").
        save_dir: Directory to save plots.
        v0, alpha, g, beta: Physical parameters.
    """
    from src.projectile_trajectory.generate_dataset import _flight_time
    T_flight = _flight_time(case, v0=v0, alpha=alpha, g=g, beta=beta)
    t = np.linspace(0, T_flight, 201)

    x_exact, y_exact, vx_exact, vy_exact, _, _ = exact_solution(
        t, case, v0=v0, alpha=alpha, g=g, beta=beta,
    )
    x_pred, y_pred, vx_pred, vy_pred = compute_trajectory_derivatives(model, t)

    speed_exact = np.sqrt(vx_exact**2 + vy_exact**2)
    speed_pred = np.sqrt(vx_pred**2 + vy_pred**2)

    case_title = case.replace("_", " ").title()

    # 1. Trajectory (y vs x) -- the hero plot
    plot_line(
        x=x_exact,
        ys=[y_exact, y_pred],
        labels=["Exact", "PINN"],
        title=f"Trajectory — {case_title}",
        xlabel=r"$x$ [m]",
        ylabel=r"$y$ [m]",
        save_path=f"{save_dir}/projectile_trajectory_{case}.png",
        colors=[COLOR_EXACT, COLOR_PINN],
    )

    # 2. Position vs time
    with pinn_style():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(t, x_exact, label="Exact", color=COLOR_EXACT)
        ax1.plot(t, x_pred, label="PINN", color=COLOR_PINN)
        ax1.set_xlabel(r"$t$ [s]")
        ax1.set_ylabel(r"$x$ [m]")
        ax1.set_title(f"Horizontal Position — {case_title}")
        ax1.legend()

        ax2.plot(t, y_exact, label="Exact", color=COLOR_EXACT)
        ax2.plot(t, y_pred, label="PINN", color=COLOR_PINN)
        ax2.set_xlabel(r"$t$ [s]")
        ax2.set_ylabel(r"$y$ [m]")
        ax2.set_title(f"Vertical Position — {case_title}")
        ax2.legend()

        save_figure(fig, f"{save_dir}/projectile_position_{case}.png")

    # 3. Velocity vs time
    with pinn_style():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(t, vx_exact, label="Exact", color=COLOR_EXACT)
        ax1.plot(t, vx_pred, label="PINN", color=COLOR_PINN)
        ax1.set_xlabel(r"$t$ [s]")
        ax1.set_ylabel(r"$v_x$ [m/s]")
        ax1.set_title(f"Horizontal Velocity — {case_title}")
        ax1.legend()

        ax2.plot(t, vy_exact, label="Exact", color=COLOR_EXACT)
        ax2.plot(t, vy_pred, label="PINN", color=COLOR_PINN)
        ax2.set_xlabel(r"$t$ [s]")
        ax2.set_ylabel(r"$v_y$ [m/s]")
        ax2.set_title(f"Vertical Velocity — {case_title}")
        ax2.legend()

        save_figure(fig, f"{save_dir}/projectile_velocity_{case}.png")

    # 4. Speed vs time
    plot_line(
        x=t,
        ys=[speed_exact, speed_pred],
        labels=["Exact", "PINN"],
        title=f"Speed — {case_title}",
        xlabel=r"$t$ [s]",
        ylabel=r"$|\mathbf{v}|$ [m/s]",
        save_path=f"{save_dir}/projectile_speed_{case}.png",
        colors=[COLOR_EXACT, COLOR_PINN],
    )

    # 5. Absolute error
    pos_error = np.sqrt((x_exact - x_pred)**2 + (y_exact - y_pred)**2)
    plot_line(
        x=t,
        ys=[pos_error],
        labels=[r"$|\mathbf{r}_\mathrm{exact} - \mathbf{r}_\mathrm{pred}|$"],
        title=f"Position Error — {case_title}",
        xlabel=r"$t$ [s]",
        ylabel=r"$|\Delta \mathbf{r}|$ [m]",
        save_path=f"{save_dir}/projectile_error_{case}.png",
        colors=[COLOR_CYCLE[2]],
    )

    # Print verification metrics
    l2_rel_x = np.linalg.norm(x_exact - x_pred) / np.linalg.norm(x_exact)
    l2_rel_y = np.linalg.norm(y_exact - y_pred) / np.linalg.norm(y_exact)
    print(f"\n{'='*50}")
    print(f"Case: {case}")
    print(f"L2 relative error (x):     {l2_rel_x:.6e}")
    print(f"L2 relative error (y):     {l2_rel_y:.6e}")
    print(f"Max position error:        {np.max(pos_error):.6e}")
    print(f"Mean position error:       {np.mean(pos_error):.6e}")

    # IC satisfaction
    x0, y0, vx0, vy0 = compute_trajectory_derivatives(model, np.array([0.0]))
    print(f"\nIC satisfaction:")
    print(f"  x(0)   = {x0[0]:.6e}")
    print(f"  y(0)   = {y0[0]:.6e}")
    print(f"  vx(0)  = {vx0[0]:.6e}  (target: {v0 * np.cos(alpha):.4f})")
    print(f"  vy(0)  = {vy0[0]:.6e}  (target: {v0 * np.sin(alpha):.4f})")
    print(f"{'='*50}\n")


def main(epoch, case="no_drag", save_dir="src/figures"):
    """Generate plots from saved prediction files (no trained model needed)."""
    data_dir = f"./src/projectile_trajectory/data/{case}"

    scaling = np.load(f"{data_dir}/scaling.npy", allow_pickle=True).item()
    L_ref = scaling["L_ref"]
    T_ref = scaling["T_ref"]

    # t_star is saved in non-dim form; convert back to physical
    t = np.load(f"{data_dir}/t_star.npy").flatten() * T_ref
    x_exact = np.load(f"{data_dir}/x_exact.npy")
    y_exact = np.load(f"{data_dir}/y_exact.npy")

    # Predictions are in non-dim space; unscale
    xy_pred_raw = torch.load(f"{data_dir}/predictions_{epoch}.pkl")
    xy_pred = torch.cat(xy_pred_raw, dim=0).numpy()
    x_pred = xy_pred[:, 0] * L_ref
    y_pred = xy_pred[:, 1] * L_ref

    case_title = case.replace("_", " ").title()

    # Trajectory
    plot_line(
        x=x_exact,
        ys=[y_exact, y_pred],
        labels=["Exact", "PINN"],
        title=f"Trajectory — {case_title}",
        xlabel=r"$x$ [m]",
        ylabel=r"$y$ [m]",
        save_path=f"{save_dir}/projectile_trajectory_{case}.png",
        colors=[COLOR_EXACT, COLOR_PINN],
    )

    # Error
    pos_error = np.sqrt((x_exact - x_pred)**2 + (y_exact - y_pred)**2)
    plot_line(
        x=t,
        ys=[pos_error],
        labels=[r"$|\mathbf{r}_\mathrm{exact} - \mathbf{r}_\mathrm{pred}|$"],
        title=f"Position Error — {case_title}",
        xlabel=r"$t$ [s]",
        ylabel=r"$|\Delta \mathbf{r}|$ [m]",
        save_path=f"{save_dir}/projectile_error_{case}.png",
        colors=[COLOR_CYCLE[2]],
    )


if __name__ == "__main__":
    for case in ["no_drag", "linear_drag"]:
        main(8000, case=case)
