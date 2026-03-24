# PINN examples with Pytorch Lightning
This repository is a collection of continuous-time and discrete-time physics-informed neural networks (PINNs) implemented in Pytorch Lightning.


<details>
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
  - [Euler-Bernoulli Beam (continuous-time)](#euler-bernoulli-beam-continuous-time)
  - [Projectile Trajectory (continuous-time)](#projectile-trajectory-continuous-time)
  - [Visualization](#visualization)
  - [Variable Glossary](#variable-glossary)
    - [Physics / PDE](#physics--pde)
    - [Domain](#domain)
    - [Training Data](#training-data)
    - [Model Predictions](#model-predictions)
    - [Loss](#loss)
    - [Neural Network Architecture](#neural-network-architecture)
    - [Training Configuration](#training-configuration)
    - [FDM (Finite Difference Method) — Heat 2D ground truth](#fdm-finite-difference-method--heat-2d-ground-truth)
- [(Applied) Ressources for continuous-time and discrete-time PINNs](#applied-ressources-for-continuous-time-and-discrete-time-pinns)

</details>


## Raissi's Burgers Equation (continuous-time)
The **first** example is Raissi's solution to [Burgers' Equation](https://en.wikipedia.org/wiki/Burgers%27_equation):
```math
\dfrac{d u}{d t} + u \dfrac{d u}{d x} = v \dfrac{d^2 u}{d x^2}  \, .
```
The aim is to train a neural network that can be used for inference. An implementation detail is that the time domain is added via a concatenated dataset. With increasing epochs from 500 to 20000, the solution gets approximated more accurately:

![](src/figures/raissi_burgers.gif)


## Moseley's Damped Harmonic Oscillator (continuous-time)
The **second** example is Moseley's identification of the friction coefficient of the [damped harmonic oscillator](https://en.wikipedia.org/wiki/Harmonic_oscillator):
```math
m \dfrac{d^2 x}{d t^2} + \mu \dfrac{d x}{d t} + kx = 0 \, .
```
This is an **inverse problem**: given noisy observations of the displacement $x(t)$, the PINN learns the friction coefficient $\mu$ (true value $\mu = 4$) while satisfying the PDE constraint. The initial conditions are $x(0) = 1$ and $\dot{x}(0) = 0$ with parameters $\omega_0 = 20$ and $d = 2$. An implementation detail is that the time domain is added as a hyperparameter. The evolution of $\mu$ over training is tracked.

![](src/figures/moseley_oscillator_solution.png)

![](src/figures/moseley_oscillator_mu_convergence.png)


## Heat Equation 1D (continuous-time)
The **third** example solves the one-dimensional [heat equation](https://en.wikipedia.org/wiki/Heat_equation):
```math
\dfrac{d u}{d t} = \alpha \dfrac{d^2 u}{d x^2} + \sigma \, .
```
An implementation detail is the source term $\sigma$. The plot compares the analytical solution with the prediction of the PINN:

![](src/figures/heat_eq_1d_analytical_vs_pinn.png)


## Heat Equation 2D (continuous-time)
The **fourth** example solves the two-dimensional [heat equation](https://en.wikipedia.org/wiki/Heat_equation):
```math
m \dfrac{d u}{d t} = \alpha \left( \dfrac{d^2 u}{d x^2} + \dfrac{d^2 u}{d y^2} \right) \, .
```
Here, no source term $\sigma$ is present. The neural network learns to solve a rectangular plate that is exposed to heat at one or more edges. The current example implements a boundary condition at the upper and left edge. With increasing time, the heat spreads across the plate:

![](src/figures/heat_eq_2d_2BC.gif)


## Wave Equation 1D (continuous-time)
The **fifth** example solves the one-dimensional [wave equation](https://en.wikipedia.org/wiki/Wave_equation):
```math
\dfrac{d^2 u}{d t^2} = c^2 \dfrac{d^2 u}{d x^2} \, .
```
This models a vibrating string with fixed endpoints. The boundary conditions are $u(0, t) = u(L, t) = 0$ and the initial condition is a sinusoidal displacement $u(x, 0) = \sin(\pi x / L)$ with zero initial velocity. The analytical solution $u(x, t) = \sin(\pi x / L) \cos(\pi c t / L)$ is used to validate the PINN prediction via a 3-row heatmap comparison (exact, PINN, difference).

![](src/figures/wave_eq_1d_analytical_vs_pinn.png)


## Inverse Burgers Equation (continuous-time)
The **sixth** example solves the [Burgers' Equation](https://en.wikipedia.org/wiki/Burgers%27_equation) as an **inverse problem**:
```math
\dfrac{d u}{d t} + u \dfrac{d u}{d x} = \nu \dfrac{d^2 u}{d x^2} \, .
```
Unlike the forward problem, the viscosity coefficient $\nu$ is unknown and must be discovered from sparse, noisy observations. Following Raissi's two-stage approach, the PINN is first trained with Adam and then refined with L-BFGS-B (full-batch). The viscosity is parameterized in log-space ($\log \nu$) to ensure positivity and improve optimization for small parameter values. The PINN simultaneously fits the data and satisfies the PDE constraint, converging to the true value of $\nu = 0.01/\pi \approx 0.003183$. Two plots are generated: a 3-row heatmap comparison (exact vs PINN vs difference) and a line plot showing the convergence of the learned $\nu$ parameter over training steps.

![](src/figures/inverse_burgers_solution_comparison.png)

![](src/figures/inverse_burgers_nu_convergence.png)


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

![](src/figures/navier_stokes_kovasznay_u_velocity_comparison.png)

![](src/figures/navier_stokes_kovasznay_v_velocity_comparison.png)

![](src/figures/navier_stokes_kovasznay_pressure_p_comparison.png)


## Raissi's Allen-Cahn Equation (discrete-time)
The **eighth** example is Raissi's discrete-time solution to the [Allen-Cahn Equation](https://en.wikipedia.org/wiki/Allen%E2%80%93Cahn_equation):
```math
\dfrac{d u}{d t} = 0.0001 \dfrac{d^2 u}{d x^2} + 5u - 5u^3 \, .
```
This solves the PDE with only two time-snapshots at t0 and t1. The neural network outputs $q$ stages of the [Runge-Kutta](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods) method, representing the time discretization. Raissi's paper reports a relative L2 error of 6.99e-3 [PINN Part I](https://arxiv.org/pdf/1711.10561.pdf). The relative L2 error is printed and shown in the plot title after training.

![](src/figures/raissi_allen_cahn_solution_comparison.png)


## Euler-Bernoulli Beam (continuous-time)
The **ninth** example solves the [Euler-Bernoulli beam equation](https://en.wikipedia.org/wiki/Euler%E2%80%93Bernoulli_beam_theory):
```math
EI \dfrac{d^4 w}{d x^4} = q(x) \, .
```
The PINN learns the deflection $w(x)$ and derives the shear force and bending moment via automatic differentiation. Three load cases are implemented: a cantilever beam with a point load, a cantilever beam with a uniformly distributed load (UDL), and a simply supported beam with a UDL.

> **Note on higher-order derivatives:** The shear force $V = EI\,w'''$ (3rd derivative) can exhibit oscillations, particularly for the cantilever point load case where $V$ should be constant. This is a known limitation of PINNs applied to higher-order ODEs — each successive derivative amplifies neural network approximation noise.

<details>
<summary><h3>Cantilever with Point Load</h3></summary>

![](src/figures/euler_bernoulli_beam_deflection_cantilever_point_load.png)

![](src/figures/euler_bernoulli_beam_moment_cantilever_point_load.png)

![](src/figures/euler_bernoulli_beam_shear_cantilever_point_load.png)

![](src/figures/euler_bernoulli_beam_error_cantilever_point_load.png)

</details>

<details>
<summary><h3>Simply Supported with UDL</h3></summary>

![](src/figures/euler_bernoulli_beam_deflection_simply_supported_udl.png)

![](src/figures/euler_bernoulli_beam_moment_simply_supported_udl.png)

![](src/figures/euler_bernoulli_beam_shear_simply_supported_udl.png)

![](src/figures/euler_bernoulli_beam_error_simply_supported_udl.png)

</details>

<details>
<summary><h3>Cantilever with UDL</h3></summary>

![](src/figures/euler_bernoulli_beam_deflection_cantilever_udl.png)

![](src/figures/euler_bernoulli_beam_moment_cantilever_udl.png)

![](src/figures/euler_bernoulli_beam_shear_cantilever_udl.png)

![](src/figures/euler_bernoulli_beam_error_cantilever_udl.png)

</details>


## Projectile Trajectory (continuous-time)
The **tenth** example solves the equations of motion for a projectile with optional linear drag:
```math
\ddot{x} = -\beta \dot{x} \, , \qquad \ddot{y} = -g - \beta \dot{y} \, .
```
For the no-drag case ($\beta = 0$), this reduces to ideal ballistic motion with a parabolic trajectory. For the linear drag case ($\beta > 0$), the drag force is proportional to velocity, producing a shorter range and asymmetric arc. The PINN learns both position components $(x(t), y(t))$ simultaneously via a multi-output network and derives velocities via automatic differentiation. Both cases have closed-form analytical solutions for validation.

<details>
<summary><h3>No Drag</h3></summary>

![](src/figures/projectile_trajectory_no_drag.png)

![](src/figures/projectile_position_no_drag.png)

![](src/figures/projectile_velocity_no_drag.png)

![](src/figures/projectile_error_no_drag.png)

</details>

<details>
<summary><h3>Linear Drag</h3></summary>

![](src/figures/projectile_trajectory_linear_drag.png)

![](src/figures/projectile_position_linear_drag.png)

![](src/figures/projectile_velocity_linear_drag.png)

![](src/figures/projectile_error_linear_drag.png)

</details>



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
| `log_nu` | Log-space viscosity parameter (learnable, `nu = exp(log_nu)`) | Inverse Burgers |
| `mu` | Damping/friction coefficient (learnable parameter) | Moseley oscillator |
| `w0` | Natural angular frequency ω₀ | Moseley oscillator |
| `d` | Damping ratio | Moseley oscillator |
| `k` | Spring constant (= ω₀²) | Moseley oscillator |
| `sigma` | Heat source term | Heat 1D |
| `w` | Transverse beam deflection (positive downward) | Euler-Bernoulli beam |
| `theta` | Beam slope dw/dx | Euler-Bernoulli beam |
| `M` | Bending moment EI · d²w/dx² | Euler-Bernoulli beam |
| `V` | Shear force EI · d³w/dx³ | Euler-Bernoulli beam |
| `EI` | Flexural rigidity (Young's modulus × second moment of area) | Euler-Bernoulli beam |
| `P` | Point load magnitude | Euler-Bernoulli beam |
| `q` | Distributed load intensity / Number of Runge-Kutta stages | Euler-Bernoulli beam / Allen-Cahn |
| `bc_orders` | Derivative order for BCs (0=w, 1=θ, 2=M, 3=V) | Euler-Bernoulli beam |
| `v0` | Initial projectile speed | Projectile trajectory |
| `alpha` | Launch angle (radians) / Thermal diffusivity | Projectile trajectory / Heat 1D/2D |
| `beta` | Linear drag coefficient (b/m) | Projectile trajectory |
| `g` | Gravitational acceleration | Projectile trajectory |
| `x_t`, `y_t` | Velocity components dx/dt, dy/dt | Projectile trajectory |
| `x_tt`, `y_tt` | Acceleration components d²x/dt², d²y/dt² | Projectile trajectory |
| `ic_orders` | Derivative order for ICs (0=position, 1=velocity) | Projectile trajectory |
| `ic_output_indices` | Output index for ICs (0=x, 1=y) | Projectile trajectory |
| `r` | Radial coordinate | Thick-walled cylinder |
| `u_r` | Radial derivative du/dr | Thick-walled cylinder |
| `u_rr` | Second radial derivative d²u/dr² | Thick-walled cylinder |
| `sigma_r` | Radial stress | Thick-walled cylinder |
| `sigma_theta` | Hoop (tangential) stress | Thick-walled cylinder |
| `sigma_vm` | Von Mises equivalent stress | Thick-walled cylinder |
| `a` | Inner radius of cylinder | Thick-walled cylinder |
| `b` | Outer radius of cylinder | Thick-walled cylinder |
| `nu_poisson` | Poisson's ratio (named to avoid clash with viscosity `nu`) | Thick-walled cylinder |
| `p_i` | Internal pressure | Thick-walled cylinder |
| `p_o` | External pressure | Thick-walled cylinder |
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
| `x_train_BC` | Boundary condition input coordinates (with derivative order column for beam) | Heat 2D, Allen-Cahn, Euler-Bernoulli beam |
| `y_train_BC` | Boundary condition target values | Euler-Bernoulli beam |
| `y_train_IC` | Initial condition target values | Allen-Cahn |
| `u_exact` | Exact/reference solution over the full domain | Heat 1D, Burgers |
| `u_exact_IC` | Exact solution at initial condition | Heat 1D, Burgers |
| `u_exact_BC_lb` | Exact solution at lower boundary | Heat 1D, Burgers |
| `u_exact_BC_ub` | Exact solution at upper boundary | Burgers |
| `x_star` | Full test grid coordinates for evaluation/prediction | All examples |
| `n_bc_points` | Number of sampled boundary/IC training points | Heat 1D, Burgers |
| `n_ic_points` | Number of sampled initial condition points | Allen-Cahn |
| `n_collocation_points` | Number of collocation points for PDE residual | Burgers |
| `n_observation_points` | Number of sparse observation points | Inverse Burgers |
| `noise_level` | Standard deviation of observation noise | Inverse Burgers |
| `x_obs` | Sparse observation point coordinates | Inverse Burgers |
| `u_obs` | Noisy observation target values | Inverse Burgers |
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
| `loss_obs` | Observation fitting loss (sparse noisy data) | Inverse Burgers |
| `loss_PDE_param` | Weighting coefficient for PDE loss | Heat 1D/2D, Burgers |
| `loss_IC_param` | Weighting coefficient for IC loss | Allen-Cahn |
| `loss_obs_param` | Weighting coefficient for observation loss | Inverse Burgers |
| `loss_BC_param` | Weighting coefficient for BC loss | Euler-Bernoulli beam |

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
| `nus` | List tracking `nu` evolution across training steps | Inverse Burgers |
| `nu_initial` | Initial guess for viscosity parameter | Inverse Burgers |

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
- https://github.com/benmoseley/harmonic-oscillator-pinn
- https://www.youtube.com/watch?v=GWjnFVIGwIg&list=PLJkYEExhe7rYY5HjpIJbgo-tDZ3bIAqAm&index=3

**Prateek Bhustali** (TU Delft)
- https://github.com/omniscientoctopus/Physics-Informed-Neural-Networks

**Ian Henderson** (University of Toulouse)
- https://towardsdatascience.com/physics-informed-neural-networks-pinns-an-intuitive-guide-fff138069563

**Daniel Crews**
- https://github.com/crewsdw/pinns_project

**Jay Roxis**
- https://github.com/jayroxis/PINNs
