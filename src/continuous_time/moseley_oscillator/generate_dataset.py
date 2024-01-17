""" Generate Moseley's dataset.
"""
import pandas as pd
import torch
import numpy as np


def generate_dataset(path: str = "./src/continuous_time/moseley_oscillator/data/dataset.parquet",):
    def exact_solution(d, w0, t):
        assert d < w0
        w = np.sqrt(w0 ** 2 - d ** 2)
        phi = np.arctan(-d/w)
        A = 1 / (2 * np.cos(phi))
        cos = torch.cos(phi + w * t)
        exp = torch.exp(-d * t)
        u = exp * 2 * A * cos
        return u
    
    d, w0 = 2, 20
    _, k = 2 * d, w0 ** 2

    t_obs = torch.rand(40).view(-1,1)
    u_obs = exact_solution(d, w0, t_obs) + 0.04 * torch.randn_like(t_obs)

    df = pd.DataFrame(data=np.concatenate([t_obs, u_obs], axis=1), columns=["t_obs", "u_obs"])
    df.to_parquet(path)


if __name__ == "__main__":
    generate_dataset()
