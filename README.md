# PINN examples with Pytorch Lightning
This repository is a collection of continuous-time and discrete-time physics-informed neural networks (PINNs) implemented in Pytorch Lightning.


<summary><h2>Table of Contents</h2></summary>

- [PINN examples with Pytorch Lightning](#pinn-examples-with-pytorch-lightning)
  - [Raissi's Burgers Equation (continuous-time)](#raissis-burgers-equation-continuous-time)
  - [Moseley's Damped Harmonic Oscillator (continuous-time)](#moseleys-damped-harmonic-oscillator-continuous-time)
  - [Heat Equation 1D (continuous-time)](#heat-equation-1d-continuous-time)
  - [Heat Equation 2D (continuous-time)](#heat-equation-2d-continuous-time)
  - [Wave Equation 1D (continuous-time)](#wave-equation-1d-continuous-time)
  - [Inverse Burgers Equation (continuous-time)](#inverse-burgers-equation-continuous-time)
  - [Navier-Stokes Kovasznay Flow (continuous-time)](#navier-stokes-kovasznay-flow-continuous-time)
  - [Raissi's Allen-Cahn Equation (discrete-time)](#raissis-allen-cahn-equation-discrete-time)
  - [Visualization](#visualization)
  - [Variable Glossary](#variable-glossary)
- [(Applied) Ressources for continuous-time and discrete-time PINNs](#applied-ressources-for-continuous-time-and-discrete-time-pinns)


## Raissi's Burgers Equation (continuous-time)
The **first** example is Raissi's solution to [Burgers' Equation](https://en.wikipedia.org/wiki/Burgers%27_equation):
```math
\dfrac{d u}{d t} + u \dfrac{d u}{d x} = v \dfrac{d^2 u}{d x^2}  \, .
```
The aim is to train a neural network that can be used for inference. An implementation detail is that the time domain is added via a concatenated dataset. With increasing epochs from 500 to 20000, the solution gets approximated more accurately:

![](https://github.com/JanAlexanderZak/pinn_examples/blob/main/src/raissi_burgers/raissi_burgers.gif)


## Moseley's Damped Harmonic Oscillator (continuous-time)
The **second** example is Moseley's identification the friction coefficient of the [damped harmonic oscillator](https://en.wikipedia.org/wiki/Harmonic_oscillator):
```math
m \dfrac{d^2 x}{d t^2} + \mu \dfrac{d x}{d t} + kx = 0 \, .
```
An implementation detail is that the time domain is added as a hyperparameter. In this case, the evolution of $\mu$ is of interest. The plot is not reproduced here, yet.

<img src="https://github.com/JanAlexanderZak/pinn_examples/blob/main/src/moseley_oscillator/mu_plot.png" width="250" height="250" />

## Heat Equation 1D (continuous-time)
The **third** example solves the one-dimensional [heat equation](https://en.wikipedia.org/wiki/Heat_equation):
```math
\dfrac{d u}{d t} = \alpha \dfrac{d^2 u}{d x^2} + \sigma \, .
```
An implementation detail is the source term $\sigma$. The plot compares the analytical solution with the prediction of the PINN:

![image](https://github.com/JanAlexanderZak/pinn_examples/blob/main/src/heat_eq_1d/analytical_vs_pinn.png)


## Heat Equation 2D (continuous-time)
The **fourth** example solves the two-dimensional [heat equation](https://en.wikipedia.org/wiki/Heat_equation):
```math
m \dfrac{d u}{d t} = \alpha \left( \dfrac{d^2 u}{d x^2} + \dfrac{d^2 u}{d y^2} \right) \, .
```
Here, no source term $\sigma$ is present. The neural network learns to solve a rectangular plate that is exposed to heat at one or more edges. The current example implements a boundary condition at the upper and left edge. With increasing time, the heat spreads across the plate:

![](https://github.com/JanAlexanderZak/pinn_examples/blob/main/src/heat_eq_2d/heat_eq_2d_2BC.gif)


## Wave Equation 1D (continuous-time)
The **fifth** example solves the one-dimensional [wave equation](https://en.wikipedia.org/wiki/Wave_equation):
```math
\dfrac{d^2 u}{d t^2} = c^2 \dfrac{d^2 u}{d x^2} \, .
```
This models a vibrating string with fixed endpoints. The boundary conditions are $u(0, t) = u(L, t) = 0$ and the initial condition is a sinusoidal displacement $u(x, 0) = \sin(\pi x / L)$ with zero initial velocity. The analytical solution $u(x, t) = \sin(\pi x / L) \cos(\pi c t / L)$ is used to validate the PINN prediction via a 3-row heatmap comparison (exact, PINN, difference).


## Inverse Burgers Equation (continuous-time)
The **sixth** example solves the [Burgers' Equation](https://en.wikipedia.org/wiki/Burgers%27_equation) as an **inverse problem**:
```math
\dfrac{d u}{d t} + u \dfrac{d u}{d x} = \nu \dfrac{d^2 u}{d x^2} \, .
```
Unlike the forward problem, the viscosity coefficient $\nu$ is unknown and must be discovered from sparse, noisy observations. The PINN simultaneously fits the data and satisfies the PDE constraint, converging to the true value of $\nu$. Two plots are generated: a 3-row heatmap comparison (exact vs PINN vs difference) and a line plot showing the convergence of the learned $\nu$ parameter over training steps.


## Navier-Stokes Kovasznay Flow (continuous-time)
The **seventh** example solves the 2D steady incompressible [Navier-Stokes equations](https://en.wikipedia.org/wiki/Navier%E2%80%93Stokes_equations) for [Kovasznay flow](https://en.wikipedia.org/wiki/Kovasznay_flow):
```math
u \dfrac{\partial u}{\partial x} + v \dfrac{\partial u}{\partial y} = -\dfrac{\partial p}{\partial x} + \nu \left( \dfrac{\partial^2 u}{\partial x^2} + \dfrac{\partial^2 u}{\partial y^2} \right) \, ,
```
```math
u \dfrac{\partial v}{\partial x} + v \dfrac{\partial v}{\partial y} = -\dfrac{\partial p}{\partial y} + \nu \left( \dfrac{\partial^2 v}{\partial x^2} + \dfrac{\partial^2 v}{\partial y^2} \right) \, ,
```
```math
\dfrac{\partial u}{\partial x} + \dfrac{\partial v}{\partial y} = 0 \, .
```
A single neural network outputs three fields $(u, v, p)$ — the x-velocity, y-velocity, and pressure. The Kovasznay analytical solution at $Re = 20$ provides ground truth for validation. Three separate 3-row heatmap comparisons are generated, one for each field.


## Raissi's Allen-Cahn Equation (discrete-time)
The **eighth** example is Raissi's discrete-time solution to the [Allen-Cahn Equation](https://en.wikipedia.org/wiki/Allen%E2%80%93Cahn_equation):
```math
\dfrac{d u}{d t} = 0.0001 \dfrac{d^2 u}{d x^2} + 5u - 5u^3 \, .
```
This solves the PDE with only two time-snapshots at t0 and t1. The neural network outputs $q$ stages of the [Runge-Kutta](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods) method, representing the time discretization. Raissi's paper reports an error of 0.007 [PINN Part I](https://arxiv.org/pdf/1711.10561.pdf). Here 0.0051 is achieved. Obviously, the training did not fully converge. The boundary conditions are not met.

<img src="https://github.com/JanAlexanderZak/pinn_examples/blob/main/src/raissi_allen_cahn/pred_t1.png" width="400" height="250" />

The training exhibits a high volatility. Three examplatory models are shown in the following figure. Other versions were significantly worse or did not start convergence at all.

<img src="https://github.com/JanAlexanderZak/pinn_examples/blob/main/src/raissi_allen_cahn/loss_plot.png" width="400" height="250" />


## Visualization

All plots are generated with matplotlib using a shared style module ([src/visualization.py](src/visualization.py)). The module provides:

- **`pinn_style()`** context manager for consistent font family (serif), sizes, and DPI (300) across all figures
- **`plot_heatmap_comparison()`** for the standard 3-row layout (Exact, PINN, Difference) using `viridis` for solution fields and `RdBu_r` (diverging) for error plots
- **`plot_single_heatmap()`** for standalone heatmaps with optional scatter overlays and vertical lines
- **`plot_line()`** for time series and parameter convergence plots
- **`create_gif()`** for animating training progression across epochs


## Variable Glossary

### Physics / PDE

| Variable | Meaning | Used in |
|----------|---------|---------|
| `u` | PDE solution field (temperature, velocity, or displacement) | All examples |
| `u_t` | Time derivative ∂u/∂t | Heat 1D/2D, Burgers |
| `u_x` | Spatial derivative ∂u/∂x | All continuous-time |
| `u_xx` | Second spatial derivative ∂²u/∂x² | All continuous-time |
| `u_y` | Spatial derivative ∂u/∂y | Heat 2D |
| `u_yy` | Second spatial derivative ∂²u/∂y² | Heat 2D |
| `alpha` | Thermal diffusivity coefficient | Heat 1D/2D |
| `nu` | Kinematic viscosity coefficient | Burgers |
| `mu` | Damping/friction coefficient (learnable parameter) | Moseley oscillator |
| `w0` | Natural angular frequency ω₀ | Moseley oscillator |
| `d` | Damping ratio | Moseley oscillator |
| `k` | Spring constant (= ω₀²) | Moseley oscillator |
| `sigma` | Heat source term | Heat 1D |
| `q` | Number of Runge-Kutta stages | Allen-Cahn |
| `dt` | Time step size between snapshots t₀ and t₁ | Allen-Cahn |
| `gamma` | FDM stability parameter (= α·Δt/Δx²) | Heat 2D (generate_dataset) |
| `F` | Allen-Cahn dynamics: 5u − 5u³ + 0.0001·u_xx | Allen-Cahn |
| `U0` | Reconstructed initial state via IRK integration | Allen-Cahn |

### Domain

| Variable | Meaning | Used in |
|----------|---------|---------|
| `x_domain` | Spatial grid points along x-axis | All examples |
| `y_domain` | Spatial grid points along y-axis | Heat 2D |
| `t_domain` | Temporal grid points | All examples |
| `X`, `T` | Meshgrid arrays from `x_domain` × `t_domain` | All continuous-time |
| `X`, `Y`, `T` | Meshgrid arrays from `x_domain` × `y_domain` × `t_domain` | Heat 2D |
| `lower_boundary` | Lower bounds of the computational domain | All examples |
| `upper_boundary` | Upper bounds of the computational domain | All examples |
| `x_pde_x` | PDE collocation x-coordinate (requires_grad for autograd) | Heat 1D/2D, Burgers |
| `x_pde_y` | PDE collocation y-coordinate (requires_grad for autograd) | Heat 2D |
| `x_pde_t` | PDE collocation t-coordinate (requires_grad for autograd) | Heat 1D/2D, Burgers |

### Training Data

| Variable | Meaning | Used in |
|----------|---------|---------|
| `x_train` | Collocation point coordinates (PDE domain) | All examples |
| `x_train_IC` | Initial condition input coordinates | Heat 1D, Burgers, Allen-Cahn |
| `x_train_IC_BC` | Combined IC + BC input coordinates | Heat 1D/2D, Burgers |
| `y_train_IC_BC` | Combined IC + BC target values | Heat 1D/2D, Burgers |
| `x_train_BC` | Boundary condition input coordinates | Heat 2D, Allen-Cahn |
| `y_train_IC` | Initial condition target values | Allen-Cahn |
| `u_exact` | Exact/reference solution over the full domain | Heat 1D, Burgers |
| `u_exact_IC` | Exact solution at initial condition | Heat 1D, Burgers |
| `u_exact_BC_lb` | Exact solution at lower boundary | Heat 1D, Burgers |
| `u_exact_BC_ub` | Exact solution at upper boundary | Burgers |
| `x_star` | Full test grid coordinates for evaluation/prediction | All examples |
| `n_bc_points` | Number of sampled boundary/IC training points | Heat 1D, Burgers |
| `n_ic_points` | Number of sampled initial condition points | Allen-Cahn |
| `n_collocation_points` | Number of collocation points for PDE residual | Burgers |
| `idx` | Random index array for subsampling BC/IC points | All generate_dataset |

### Model Predictions

| Variable | Meaning | Used in |
|----------|---------|---------|
| `u_pred_IC_BC` | Model prediction at combined IC/BC points | Heat 1D/2D, Burgers |
| `u_pred_PDE` | Model prediction at collocation points | Heat 1D/2D, Burgers |
| `y_pred_IC` | Model prediction at initial condition points | Allen-Cahn |
| `y_pred_IC_minus` | `y_pred_IC[:, :-1]` — first q columns (IRK stages) | Allen-Cahn |
| `y_pred_BC` | Model prediction at boundary points | Allen-Cahn |
| `u_pred` | Final prediction tensor (in predict_step / visualization) | All models |

### Loss

| Variable | Meaning | Used in |
|----------|---------|---------|
| `loss` | Total combined loss | All models |
| `loss_PDE` | PDE residual loss (MSE of physics constraint) | Heat 1D/2D, Burgers |
| `loss_IC_BC` | Combined IC + BC loss | Heat 1D/2D, Burgers |
| `loss_IC` | Initial condition loss | Allen-Cahn |
| `loss_BC` | Boundary condition loss | Allen-Cahn |
| `loss_data` | Data fitting loss (observation vs prediction) | Moseley oscillator |
| `loss_PDE_param` | Weighting coefficient for PDE loss | Heat 1D/2D, Burgers |
| `loss_IC_param` | Weighting coefficient for IC loss | Allen-Cahn |
| `loss_data_param` | Weighting coefficient for data loss | Moseley oscillator |

### Neural Network Architecture

| Variable | Meaning | Used in |
|----------|---------|---------|
| `linears` | `ModuleList` of linear layers | All models |
| `activation` | Activation function instance (e.g. `Tanh()`) | All models |
| `in_features` | Number of input features (spatial + temporal dims) | All models |
| `out_features` | Number of output features | All models |
| `num_hidden_layers` | Number of hidden layers | All models |
| `size_hidden_layers` | Number of neurons per hidden layer | All models |
| `batch_norms` | `ModuleList` of `BatchNorm1d` layers (optional) | All models |
| `dropout_layer` | `Dropout` layer (optional) | All models |
| `irk_weights` | Implicit Runge-Kutta Butcher tableau weights (101×100) | Allen-Cahn |
| `irk_times` | IRK time coefficients from Butcher tableau | Allen-Cahn |

### Training Configuration

| Variable | Meaning | Used in |
|----------|---------|---------|
| `learning_rate` | Optimizer learning rate | All executors |
| `weight_decay` | L2 regularization strength | All executors |
| `scheduler_patience` | Epochs before LR reduction | All executors |
| `scheduler_monitor` | Metric monitored by LR scheduler | All executors |
| `batch_size` | Training batch size | All executors |
| `seed` | Random seed for reproducibility | All executors |
| `mus` | List tracking `mu` evolution across epochs | Moseley oscillator |

### FDM (Finite Difference Method) — Heat 2D ground truth

| Variable | Meaning |
|----------|---------|
| `u[k, i, j]` | Temperature at time step `k`, x-index `i`, y-index `j` |
| `delta_x` | Spatial grid spacing |
| `delta_t` | Time step (derived from stability condition) |
| `gamma` | Stability parameter (must be ≤ 0.25 for 2D explicit Euler) |
| `plate_length` | Side length of the square plate |
| `max_iter_time` | Number of FDM time iterations |

---

# (Applied) Ressources for continuous-time and discrete-time PINNs

**Maziar Raissi** (Brown University)
- https://github.com/maziarraissi/PINNs

**Ameya D. Jagtap** (Brown University)
- https://github.com/AmeyaJagtap/Conservative_PINNs

**Juan Toscano** (Brown University)
- Part 1: https://www.youtube.com/watch?v=AXXnSzmpyoI
- Part 2: https://www.youtube.com/watch?v=77jChHTcbv0
- Part 3: https://www.youtube.com/watch?v=YpNYVD9B_Js
- https://github.com/jdtoscano94/Learning-Python-Physics-Informed-Machine-Learning-PINNs-DeepONets

**Ben Moseley** (University of Oxford)
- https://ora.ox.ac.uk/objects/uuid:b790477c-771f-4926-99c6-d2f9d248cb23
- https://benmoseley.blog/my-research/so-what-is-a-physics-informed-neural-network/
https://github.com/benmoseley/harmonic-oscillator-pinn
- https://www.youtube.com/watch?v=GWjnFVIGwIg&list=PLJkYEExhe7rYY5HjpIJbgo-tDZ3bIAqAm&index=3

**Prateek Bhustali** (TU Delft)
- https://github.com/omniscientoctopus/Physics-Informed-Neural-Networks

**Ian Henderson** (University of Toulouse)
- https://towardsdatascience.com/physics-informed-neural-networks-pinns-an-intuitive-guide-fff138069563

**Daniel Crews**
- https://github.com/crewsdw/pinns_project

**Jay Roxis**
- https://github.com/jayroxis/PINNs
