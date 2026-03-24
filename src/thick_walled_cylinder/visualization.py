""" Visualization for thick-walled cylinder (Lame problem) PINN results.

Produces comparison plots of displacement, stress distribution,
von Mises stress, and absolute error (exact vs. PINN prediction).
"""
import numpy as np
import torch

from src.visualization import plot_line, pinn_style, save_figure, FIGSIZE_LINE_PLOT, COLOR_EXACT, COLOR_PINN, COLOR_CYCLE
from src.thick_walled_cylinder.generate_dataset import exact_solution

import matplotlib.pyplot as plt


def compute_cylinder_derivatives(model, r_np):
    """Compute u, sigma_r, sigma_theta from the trained model using autograd.

    Args:
        model: Trained CylinderPINNRegressor.
        r_np: Numpy array of radial positions, shape (n,) or (n, 1).

    Returns:
        (u, sigma_r, sigma_theta, sigma_vm): Numpy arrays.
    """
    r_np = np.asarray(r_np).flatten()
    r_t = torch.tensor(r_np.reshape(-1, 1), dtype=torch.float32, requires_grad=True)

    E = model.hparams.E
    nu = model.hparams.nu_poisson

    u = model.net(r_t)
    ones = torch.ones_like(u)

    u_r = torch.autograd.grad(u, r_t, ones, retain_graph=True, create_graph=True)[0]

    coeff = E / (1.0 - nu**2)
    sigma_r = coeff * (u_r + nu * u / r_t)
    sigma_theta = coeff * (u / r_t + nu * u_r)

    u_np = u.detach().numpy().flatten()
    sr_np = sigma_r.detach().numpy().flatten()
    st_np = sigma_theta.detach().numpy().flatten()

    sigma_vm_np = np.sqrt(sr_np**2 - sr_np * st_np + st_np**2)

    return u_np, sr_np, st_np, sigma_vm_np


def visualize(
    model,
    load_case="internal_pressure",
    save_dir="src/figures",
    a=1.0,
    b=2.0,
    E=1.0,
    nu=0.3,
    p_i=1.0,
    p_o=0.5,
):
    """Generate all cylinder visualization plots.

    Args:
        model: Trained CylinderPINNRegressor.
        load_case: Load case string.
        save_dir: Directory to save plots.
        a, b, E, nu, p_i, p_o: Physical parameters.
    """
    r = np.linspace(a, b, 201)
    u_exact, sr_exact, st_exact, vm_exact = exact_solution(
        r, load_case, a=a, b=b, E=E, nu=nu, p_i=p_i, p_o=p_o,
    )
    u_pred, sr_pred, st_pred, vm_pred = compute_cylinder_derivatives(model, r)

    case_title = load_case.replace("_", " ").title()

    # 1. Displacement
    plot_line(
        x=r,
        ys=[u_exact, u_pred],
        labels=["Exact", "PINN"],
        title=f"Radial Displacement — {case_title}",
        xlabel=r"$r$ [-]",
        ylabel=r"$u$ [-]",
        save_path=f"{save_dir}/thick_walled_cylinder_displacement_{load_case}.png",
        colors=[COLOR_EXACT, COLOR_PINN],
    )

    # 2. Stress distribution (sigma_r and sigma_theta on same plot)
    with pinn_style():
        fig, ax = plt.subplots(figsize=FIGSIZE_LINE_PLOT)

        ax.plot(r, sr_exact, label=r"$\sigma_r$ Exact", color=COLOR_EXACT, linestyle="-")
        ax.plot(r, st_exact, label=r"$\sigma_\theta$ Exact", color=COLOR_EXACT, linestyle="--")
        ax.plot(r, sr_pred, label=r"$\sigma_r$ PINN", color=COLOR_PINN, linestyle="-")
        ax.plot(r, st_pred, label=r"$\sigma_\theta$ PINN", color=COLOR_CYCLE[0], linestyle="-")

        ax.axhline(y=0, color="gray", linestyle=":", linewidth=0.5)
        ax.set_xlabel(r"$r$ [-]")
        ax.set_ylabel(r"$\sigma$ [-]")
        ax.set_title(f"Stress Distribution — {case_title}")
        ax.legend()

        save_figure(fig, f"{save_dir}/thick_walled_cylinder_stress_{load_case}.png")

    # 3. Von Mises stress
    plot_line(
        x=r,
        ys=[vm_exact, vm_pred],
        labels=["Exact", "PINN"],
        title=f"Von Mises Stress — {case_title}",
        xlabel=r"$r$ [-]",
        ylabel=r"$\sigma_\mathrm{vm}$ [-]",
        save_path=f"{save_dir}/thick_walled_cylinder_von_mises_{load_case}.png",
        colors=[COLOR_EXACT, COLOR_PINN],
    )

    # 4. Absolute error
    abs_error = np.abs(u_exact - u_pred)
    plot_line(
        x=r,
        ys=[abs_error],
        labels=[r"$|u_\mathrm{exact} - u_\mathrm{pred}|$"],
        title=f"Displacement Error — {case_title}",
        xlabel=r"$r$ [-]",
        ylabel=r"$|\Delta u|$ [-]",
        save_path=f"{save_dir}/thick_walled_cylinder_error_{load_case}.png",
        colors=[COLOR_CYCLE[2]],
    )

    # Print verification metrics
    l2_rel = np.linalg.norm(u_exact - u_pred) / np.linalg.norm(u_exact)
    print(f"\n{'='*50}")
    print(f"Load case: {load_case}")
    print(f"L2 relative error (u):     {l2_rel:.6e}")
    print(f"Max absolute error (u):    {np.max(abs_error):.6e}")
    print(f"Mean absolute error (u):   {np.mean(abs_error):.6e}")

    # BC satisfaction (stress at boundaries)
    u_a, sr_a, _, _ = compute_cylinder_derivatives(model, np.array([a]))
    u_b, sr_b, _, _ = compute_cylinder_derivatives(model, np.array([b]))
    print(f"\nBC satisfaction:")
    print(f"  sigma_r(a={a}) = {sr_a[0]:.6e}")
    print(f"  sigma_r(b={b}) = {sr_b[0]:.6e}")
    print(f"{'='*50}\n")


def main(epoch, load_case="internal_pressure", save_dir="src/figures"):
    """Generate plots from saved data files (no trained model needed)."""
    data_dir = f"./src/thick_walled_cylinder/data/{load_case}"

    r = np.load(f"{data_dir}/r_star.npy").flatten()
    u_exact = np.load(f"{data_dir}/u_exact.npy")
    sr_exact = np.load(f"{data_dir}/sigma_r_exact.npy")
    st_exact = np.load(f"{data_dir}/sigma_theta_exact.npy")

    u_pred_raw = torch.load(f"{data_dir}/predictions_{epoch}.pkl")
    u_pred = torch.cat(u_pred_raw, dim=0).numpy().flatten()

    case_title = load_case.replace("_", " ").title()

    # Displacement
    plot_line(
        x=r,
        ys=[u_exact, u_pred],
        labels=["Exact", "PINN"],
        title=f"Radial Displacement — {case_title}",
        xlabel=r"$r$ [-]",
        ylabel=r"$u$ [-]",
        save_path=f"{save_dir}/thick_walled_cylinder_displacement_{load_case}.png",
        colors=[COLOR_EXACT, COLOR_PINN],
    )

    # Error
    abs_error = np.abs(u_exact - u_pred)
    plot_line(
        x=r,
        ys=[abs_error],
        labels=[r"$|u_\mathrm{exact} - u_\mathrm{pred}|$"],
        title=f"Displacement Error — {case_title}",
        xlabel=r"$r$ [-]",
        ylabel=r"$|\Delta u|$ [-]",
        save_path=f"{save_dir}/thick_walled_cylinder_error_{load_case}.png",
        colors=[COLOR_CYCLE[2]],
    )


if __name__ == "__main__":
    for case in ["internal_pressure", "external_pressure", "combined_pressure"]:
        main(12000, load_case=case)
