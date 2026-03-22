import torch

from src.visualization import plot_line


def main(model, epoch, save_dir="src/figures"):
    """Plot the PINN solution at a given training epoch."""
    t_test = torch.linspace(0, 1, 300).view(-1, 1)
    y_pred = model.forward(t_test).detach()

    plot_line(
        x=t_test[:, 0].numpy(),
        ys=[y_pred[:, 0].numpy()],
        labels=["PINN solution"],
        title=f"Training step {epoch}",
        xlabel="t",
        ylabel="x(t)",
        save_path=f"{save_dir}/moseley_oscillator_solution.png",
        colors=["tab:green"],
    )
