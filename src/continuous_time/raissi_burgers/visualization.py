import matplotlib.pyplot as plt
import torch
import scipy
import numpy as np

from pyDOE import lhs
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata


def visualize(index, epoch: int = 20000):

    np.random.seed(6020)

    N_u = 100
    collocation_points = 10000

    data = scipy.io.loadmat("src/continuous_time/raissi_burgers/data/burgers_shock.mat")
    t_domain = data["t"].flatten().reshape(-1, 1)  # (100, 1)
    x_domain = data["x"].flatten().reshape(-1, 1)  # (256, 1)
    y_true = np.real(data["usol"]).T

    X, T = np.meshgrid(x_domain, t_domain)  # (100, 256), (100, 256)
    X_star = np.hstack((
        X.flatten().reshape(-1, 1),
        T.flatten().reshape(-1, 1),
    ))  # (256*100=25600, 1)
    y_true_flat = y_true.flatten().reshape(-1, 1)  # (256*100=25600, 1)

    # * Boundary conditions
    lower_boundary = X_star.min(axis=0)
    upper_boundary = X_star.max(axis=0)

    xx1 = np.hstack((X[0:1, :].T, T[0:1, :].T))     # (256, 2)
    uu1 = y_true[0:1, :].T                          # (256, 1)
    xx2 = np.hstack((X[:, 0:1], T[:, 0:1]))         # (100, 2)
    uu2 = y_true[:, 0:1]                            # (100, 1)
    xx3 = np.hstack((X[:, -1:], T[:, -1:]))         # (100, 2)
    uu3 = y_true[:, -1:]                            # (100, 1)

    all_X_u_train = np.vstack([xx1, xx2, xx3])
    all_u_train = np.vstack([uu1, uu2, uu3])

    # * Sample collocation
    X_f_train = lower_boundary + (upper_boundary - lower_boundary) * lhs(2, collocation_points)
    X_f_train = np.vstack((X_f_train, all_X_u_train))  # (10000+456=10456, 2)

    # * Final data
    idx = np.random.choice(all_X_u_train.shape[0], N_u, replace=False)
    X_u_train = all_X_u_train[idx, :]   # (100, 2)
    u_train = all_u_train[idx, :]       # (100, 1)

    # Predictions
    u_pred = torch.load(f"./src/continuous_time/raissi_burgers/data/predictions/predictions_{epoch}.pkl")
    u_pred = torch.tensor(u_pred).reshape(-1, 1)
    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')

    # * Visualize
    fig = plt.figure(figsize=(9, 5))

    ax = fig.add_subplot(111)

    h = ax.imshow(
        U_pred.T,
        interpolation='nearest',
        cmap='rainbow',
        vmin=-1,
        vmax=1,
        extent=[
            t_domain.min(),
            t_domain.max(),
            x_domain.min(),
            x_domain.max(),
        ], 
        origin='lower',
        aspect='auto',
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=15) 

    ax.plot(
        X_u_train[:, 1], 
        X_u_train[:, 0], 
        'kx', label = 'Data (%d points)' % (u_train.shape[0]), 
        markersize = 4,  # marker size doubled
        clip_on = False,
        alpha=1.0
    )

    line = np.linspace(x_domain.min(), x_domain.max(), 2)[:,None]
    ax.plot(t_domain[25]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t_domain[50]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t_domain[75]*np.ones((2,1)), line, 'w-', linewidth = 1)

    ax.set_xlabel('$t$', size=20)
    ax.set_ylabel('$x$', size=20)
    ax.legend(
        loc='upper center', 
        bbox_to_anchor=(0.9, -0.05), 
        ncol=5, 
        frameon=False, 
        prop={'size': 15}
    )
    ax.set_title(f'$u(x,t)$', fontsize = 20) # font size doubled
    ax.tick_params(labelsize=15)

    import string
    alphabet = list(string.ascii_lowercase)

    fig.savefig(f"src/continuous_time/raissi_burgers/plots/epochs/{epoch}.png")

if __name__ == "__main__":
    
    for index, epoch in enumerate(np.arange(500, 20500, 500)):
        visualize(index, epoch)