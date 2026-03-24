""" Visualization for Euler-Bernoulli beam PINN results.

Produces comparison plots of deflection, moment, and shear
(exact vs. PINN prediction), plus an absolute error plot.
"""
import numpy as np
import torch

from src.visualization import plot_line, pinn_style, save_figure, FIGSIZE_LINE_PLOT, COLOR_EXACT, COLOR_PINN, COLOR_CYCLE
from src.euler_bernoulli_beam.generate_dataset import exact_solution

import matplotlib.pyplot as plt


def compute_beam_derivatives(model, x_np):
    """Compute w, theta, M, V from the trained model using autograd.

    Args:
        model: Trained BeamPINNRegressor.
        x_np: Numpy array of x positions, shape (n,) or (n, 1).

    Returns:
        (w, theta, M, V): Numpy arrays.
    """
    x_np = np.asarray(x_np).flatten()
    x_t = torch.tensor(x_np.reshape(-1, 1), dtype=torch.float32, requires_grad=True)

    EI = model.hparams.EI

    w = model.net(x_t)
    ones = torch.ones_like(w)

    w_x = torch.autograd.grad(w, x_t, ones, retain_graph=True, create_graph=True)[0]
    w_xx = torch.autograd.grad(w_x, x_t, ones, retain_graph=True, create_graph=True)[0]
    w_xxx = torch.autograd.grad(w_xx, x_t, ones, retain_graph=True, create_graph=True)[0]

    w_np = w.detach().numpy().flatten()
    theta_np = w_x.detach().numpy().flatten()
    M_np = (EI * w_xx).detach().numpy().flatten()
    V_np = (EI * w_xxx).detach().numpy().flatten()

    return w_np, theta_np, M_np, V_np


def visualize(
    model,
    load_case="cantilever_point_load",
    save_dir="src/figures",
    L=1.0,
    EI=1.0,
    P=1.0,
    q0=1.0,
):
    """Generate all beam visualization plots.

    Args:
        model: Trained BeamPINNRegressor.
        load_case: Load case string.
        save_dir: Directory to save plots.
        L, EI, P, q0: Physical parameters.
    """
    x = np.linspace(0, L, 201)
    w_exact, theta_exact, M_exact, _ = exact_solution(
        x, load_case, L=L, EI=EI, P=P, q0=q0,
    )
    w_pred, theta_pred, M_pred, _ = compute_beam_derivatives(model, x)

    case_title = load_case.replace("_", " ").title()

    # 1. Deflection
    plot_line(
        x=x,
        ys=[w_exact, w_pred],
        labels=["Exact", "PINN"],
        title=f"Deflection — {case_title}",
        xlabel=r"$x$ [-]",
        ylabel=r"$w$ [-]",
        save_path=f"{save_dir}/euler_bernoulli_beam_deflection_{load_case}.png",
        colors=[COLOR_EXACT, COLOR_PINN],
    )

    # 2. Bending moment
    plot_line(
        x=x,
        ys=[M_exact, M_pred],
        labels=["Exact", "PINN"],
        title=f"Bending Moment — {case_title}",
        xlabel=r"$x$ [-]",
        ylabel=r"$M$ [-]",
        save_path=f"{save_dir}/euler_bernoulli_beam_moment_{load_case}.png",
        colors=[COLOR_EXACT, COLOR_CYCLE[0]],
    )

    # 3. Absolute error
    abs_error = np.abs(w_exact - w_pred)
    plot_line(
        x=x,
        ys=[abs_error],
        labels=[r"$|w_\mathrm{exact} - w_\mathrm{pred}|$"],
        title=f"Absolute Error — {case_title}",
        xlabel=r"$x$ [-]",
        ylabel=r"$|\Delta w|$ [-]",
        save_path=f"{save_dir}/euler_bernoulli_beam_error_{load_case}.png",
        colors=[COLOR_CYCLE[2]],
    )

    # Print verification metrics
    l2_rel = np.linalg.norm(w_exact - w_pred) / np.linalg.norm(w_exact)
    print(f"\n{'='*50}")
    print(f"Load case: {load_case}")
    print(f"L2 relative error (w):     {l2_rel:.6e}")
    print(f"Max absolute error (w):    {np.max(abs_error):.6e}")
    print(f"Mean absolute error (w):   {np.mean(abs_error):.6e}")

    # BC satisfaction
    w0, theta0, M0, V0 = compute_beam_derivatives(model, np.array([0.0]))
    wL, thetaL, ML, VL = compute_beam_derivatives(model, np.array([L]))
    print(f"\nBC satisfaction:")
    print(f"  w(0)      = {w0[0]:.6e}")
    print(f"  theta(0)  = {theta0[0]:.6e}")
    print(f"  w(L)      = {wL[0]:.6e}")
    print(f"  M(L)      = {ML[0]:.6e}")
    print(f"  V(L)      = {VL[0]:.6e}")
    print(f"{'='*50}\n")


def main(epoch, load_case="cantilever_point_load", save_dir="src/figures"):
    """Generate beam plots from saved data files (no trained model needed)."""
    data_dir = f"./src/euler_bernoulli_beam/data/{load_case}"

    x = np.load(f"{data_dir}/x_star.npy").flatten()
    w_exact = np.load(f"{data_dir}/w_exact.npy")
    M_exact = np.load(f"{data_dir}/M_exact.npy")


    w_pred_raw = torch.load(f"{data_dir}/predictions_{epoch}.pkl")
    w_pred = torch.cat(w_pred_raw, dim=0).numpy().flatten()

    case_title = load_case.replace("_", " ").title()

    # 1. Deflection
    plot_line(
        x=x,
        ys=[w_exact, w_pred],
        labels=["Exact", "PINN"],
        title=f"Deflection — {case_title}",
        xlabel=r"$x$ [-]",
        ylabel=r"$w$ [-]",
        save_path=f"{save_dir}/euler_bernoulli_beam_deflection_{load_case}.png",
        colors=[COLOR_EXACT, COLOR_PINN],
    )

    # 2. Absolute error
    abs_error = np.abs(w_exact - w_pred)
    plot_line(
        x=x,
        ys=[abs_error],
        labels=[r"$|w_\mathrm{exact} - w_\mathrm{pred}|$"],
        title=f"Absolute Error — {case_title}",
        xlabel=r"$x$ [-]",
        ylabel=r"$|\Delta w|$ [-]",
        save_path=f"{save_dir}/euler_bernoulli_beam_error_{load_case}.png",
        colors=[COLOR_CYCLE[2]],
    )


if __name__ == "__main__":
    for case in ["cantilever_point_load", "simply_supported_udl", "cantilever_udl"]:
        main(15000, load_case=case)
