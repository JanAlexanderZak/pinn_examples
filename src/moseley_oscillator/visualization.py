import numpy as np
import torch

from src.visualization import plot_line, COLOR_PINN, COLOR_EXACT, COLOR_CYCLE


def exact_solution(d, w0, t):
    """Analytical solution of the underdamped harmonic oscillator."""
    w = np.sqrt(w0 ** 2 - d ** 2)
    phi = np.arctan(-d / w)
    A = 1 / (2 * np.cos(phi))
    u = np.exp(-d * t) * 2 * A * np.cos(phi + w * t)
    return u


def main(model, epoch, save_dir="src/figures"):
    """Plot the PINN solution vs exact solution at a given training epoch."""
    t_test = torch.linspace(0, 1, 300).view(-1, 1)
    y_pred = model.forward(t_test).detach()

    d = model.hparams.d
    w0 = model.hparams.w0
    t_np = t_test[:, 0].numpy()
    u_exact = exact_solution(d, w0, t_np)

    plot_line(
        x=t_np,
        ys=[u_exact, y_pred[:, 0].numpy()],
        labels=["Exact", "PINN"],
        title=f"Training step {epoch}",
        xlabel=r"$t$ [-]",
        ylabel=r"$x$ [-]",
        save_path=f"{save_dir}/moseley_oscillator_solution.png",
        colors=[COLOR_EXACT, COLOR_PINN],
    )

    if hasattr(model, "mus") and model.mus:
        mu_true = 2 * d  # mu = 2 * d * m, with m = 1
        plot_line(
            x=np.arange(len(model.mus)),
            ys=[np.array(model.mus)],
            labels=[r"Learned $\mu$"],
            title=r"Convergence of $\mu$",
            xlabel="Epoch",
            ylabel=r"$\mu$",
            save_path=f"{save_dir}/moseley_oscillator_mu_convergence.png",
            colors=[COLOR_CYCLE[0]],
            hlines=[(mu_true, rf"True $\mu = {mu_true}$", COLOR_EXACT)],
        )
