import torch
import numpy as np
import scipy

from src.visualization import plot_line, COLOR_EXACT, COLOR_PINN


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

    # Relative L2 error
    l2_error = np.linalg.norm(u_exact_t1 - u_pred_t1) / np.linalg.norm(u_exact_t1)
    print(f"Allen-Cahn relative L2 error at t1: {l2_error:.6f}")

    plot_line(
        x=x_domain,
        ys=[u_exact_t1, u_pred_t1],
        labels=["Exact", "PINN"],
        title=f"Allen-Cahn at t={t_domain[idx_t1]:.2f} (L2 error: {l2_error:.4f})",
        xlabel=r"$x$ [-]",
        ylabel=r"$u$ [-]",
        save_path=f"{save_dir}/raissi_allen_cahn_solution_comparison.png",
        colors=[COLOR_EXACT, COLOR_PINN],
    )


if __name__ == "__main__":
    main("0.0001_100_4_200")
