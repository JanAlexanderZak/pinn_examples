import os

import numpy as np
import scipy


def generate_dataset(path: str = "src/discrete_time/raissi_allen_cahn/data",):

    q = 100
    lower_boundary = np.array([-1.0])
    upper_boundary = np.array([1.0])
    
    N = 200
    
    data = scipy.io.loadmat("src/discrete_time/raissi_allen_cahn/data/AC.mat")
    
    t_domain = data["tt"].flatten().reshape(-1, 1)   # T x 1
    x_domain = data["x"].flatten().reshape(-1, 1)    # N x 1
    
    y_train = np.real(data["uu"]).T     # T x N
    
    idx_t0 = 20
    idx_t1 = 180
    dt = t_domain[idx_t1] - t_domain[idx_t0]
    print(dt)
    
    # Initial data, former x0, u0
    noise_u0 = 0.0
    idx_x = np.random.choice(y_train.shape[1], N, replace=False) 
    x_train_IC = x_domain[idx_x, :]
    y_train_IC = y_train[idx_t0:idx_t0 + 1, idx_x].T
    y_train_IC = y_train_IC + noise_u0 * np.std(y_train_IC) * np.random.randn(y_train_IC.shape[0], y_train_IC.shape[1])
       
    # Boundary data, former x1
    x_train_BC = np.vstack((lower_boundary, upper_boundary))
    
    # Test data
    X_star = x_domain

    # * Save
    np.save(os.path.join(path, "x_train_BC"), x_train_BC)
    np.save(os.path.join(path, "x_train_IC"), x_train_IC)
    np.save(os.path.join(path, "y_train_IC"), y_train_IC)
    np.save(os.path.join(path, "y_train"), y_train)
    np.save(os.path.join(path, "X_star"), X_star)


if __name__ == "__main__":
    generate_dataset()
