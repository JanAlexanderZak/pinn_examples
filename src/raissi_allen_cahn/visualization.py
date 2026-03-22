import torch
import numpy as np
import scipy

from src.visualization import plot_line


def main(epoch, save_dir="src/figures"):
    data = scipy.io.loadmat("src/raissi_allen_cahn/data/AC.mat")
    t_domain = data["tt"].flatten()
    x_domain = data["x"].flatten()
    u_exact = np.real(data["uu"]).T  # (T, N)

    idx_t1 = 180

    # Exact solution at t1
    u_exact_t1 = u_exact[idx_t1, :]

    # Load PINN predictions (output shape: N x 101, last column = prediction at t1)
    u_pred_raw = torch.load(
        f"./src/raissi_allen_cahn/data/predictions/predictions_{epoch}.pkl"
    )
    u_pred_raw = torch.cat(u_pred_raw, dim=0).numpy()
    u_pred_t1 = u_pred_raw[:, -1]

    plot_line(
        x=x_domain,
        ys=[u_exact_t1, u_pred_t1],
        labels=["Exact", "PINN"],
        title=f"Allen-Cahn at t={t_domain[idx_t1]:.2f}",
        xlabel="x",
        ylabel="u(x)",
        save_path=f"{save_dir}/raissi_allen_cahn_solution_comparison.png",
        colors=["black", "tab:red"],
    )


if __name__ == "__main__":
    main("0.0001_100_4_200")
