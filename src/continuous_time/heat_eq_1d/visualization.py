import torch
import numpy as np
import plotly.express as px

from scipy.interpolate import griddata
from plotly.subplots import make_subplots

from src.continuous_time.heat_eq_1d.generate_dataset import exact_solution


def main(epoch):
    # Repeat dataset generation
    lx = 1.
    nx = 21

    x_domain = np.linspace(0., lx, nx)
    t_domain = np.linspace(0, 5, 100)

    # * Initial and boundary conditions
    X, T = np.meshgrid(x_domain, t_domain)  # (100, 21), (100, 21)

    X_star = np.hstack((
        X.flatten().reshape(-1, 1),
        T.flatten().reshape(-1, 1),
    ))

    u_list = []
    for time in t_domain:
        u = exact_solution(x=x_domain, t=time, alpha=0.1)
        u_list.append(u)

    y_train = np.vstack(u_list)

    fig = make_subplots(rows=3, cols=1, vertical_spacing=0.05)
    fig = fig.add_trace(px.imshow(y_train.T, origin="lower").data[0], row=1, col=1)

    # * Prediction
    u_pred = torch.load(f"./src/continuous_time/heat_eq_1d/data/predictions/predictions_{epoch}.pkl")
    u_pred = torch.tensor(u_pred).reshape(-1, 1)
    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method="cubic")

    fig = fig.add_trace(px.imshow(U_pred.T, origin="lower").data[0], row=2, col=1)

    # Difference
    fig = fig.add_trace(px.imshow((y_train.T - U_pred.T), origin="lower").data[0], row=3, col=1)

    layout = px.imshow(U_pred).layout
    fig.layout.coloraxis = layout.coloraxis

    fig = fig.update_layout(
        xaxis=dict(showticklabels=False),
        xaxis2=dict(showticklabels=False),
        xaxis3=dict(
            title="t domain",
            tickmode="array",
            tickvals=[0, 50, 99],
            ticktext=[0, 2.5, 5],
        ),
        yaxis=dict(
            title="x domain (analytical)",
            tickmode="array",
            tickvals=[0, 10, 20],
            ticktext=[0, 0.5, 1],
        ),
        yaxis2=dict(
            title="x domain (PINN)",
            tickmode="array",
            tickvals=[0, 10, 20],
            ticktext=[0, 0.5, 1],
        ),
        yaxis3=dict(
            title="x domain (difference)",
            tickmode="array",
            tickvals=[0, 10, 20],
            ticktext=[0, 0.5, 1],
        ),
        coloraxis=dict(colorbar=dict(title="T(x,t)")),
    )
    fig.show()
    fig.write_image("src/continuous_time/heat_eq_1d/plots/analytical_solution_vs_pinn.png")


if __name__ == "__main__":
    main(10000)
