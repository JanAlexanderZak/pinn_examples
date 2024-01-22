# PINN examples with Pytorch Lightning
This repository is a collection of continuous-time and discrete-time physics-informed neural networks (PINNs) implemented in Pytorch Lightning.


<summary><h2>Table of Contents</h2></summary>

- [PINN examples with Pytorch Lightning](#pinn-examples-with-pytorch-lightning)
  - [Raissi's Burgers Equation (continuous-time)](#raissis-burgers-equation-continuous-time)
  - [Moseley's Damped Harmonic Oscillator (continuous-time)](#moseleys-damped-harmonic-oscillator-continuous-time)
  - [Heat Equation 1D (continuous-time)](#heat-equation-1d-continuous-time)
  - [Heat Equation 2D (continuous-time)](#heat-equation-2d-continuous-time)
  - [Raissi's Burgers Equation (discrete-time)](#raissis-burgers-equation-discrete-time)
- [(Applied) Ressources for continuous-time and discrete-time PINNs](#applied-ressources-for-continuous-time-and-discrete-time-pinns)


## Raissi's Burgers Equation (continuous-time)  
The **first** example is Raissi's solution to [Burgers' Equation](https://en.wikipedia.org/wiki/Burgers%27_equation):
```math
\dfrac{d u}{d t} + u \dfrac{d u}{d x} = v \dfrac{d^2 u}{d x^2}  \, .
```
The aim is to train a neural network that can be used for inference. An implementation detail is that the time domain is added via a concatenated dataset. With increasing epochs from 500 to 20000, the solution gets approximated more accurately:  

![](https://github.com/JanAlexanderZak/pinn_examples/blob/main/src/continuous_time/raissi_burgers/raissi_burgers.gif)  


## Moseley's Damped Harmonic Oscillator (continuous-time)  
The **second** example is Moseley's identification the friction coefficient of the [damped harmonic oscillator](https://en.wikipedia.org/wiki/Harmonic_oscillator):
```math
m \dfrac{d^2 x}{d t^2} + \mu \dfrac{d x}{d t} + kx = 0 \, .
```
An implementation detail is that the time domain is added as a hyperparameter. In this case, the evolution of $\mu$ is of interest. The plot is not reproduced here.  

<img src="https://github.com/JanAlexanderZak/pinn_examples/blob/main/src/continuous_time/moseley_oscillator/mu_plot.png" width="250" height="250" />

## Heat Equation 1D (continuous-time)  
The **third** example solves the one-dimensional [heat equation](https://en.wikipedia.org/wiki/Heat_equation):
```math
m \dfrac{d u}{d t} = \alpha \dfrac{d^2 u}{d x^2} + \sigma \, .
```
An implementation detail is the source term $\sigma$. The plot compares the analytical solution with the prediction of the PINN:  

![image](https://github.com/JanAlexanderZak/pinn_examples/blob/main/src/continuous_time/heat_eq_1d/analytical_vs_pinn.png)  


## Heat Equation 2D (continuous-time)  
The **fourth** example solves the two-dimensional [heat equation](https://en.wikipedia.org/wiki/Heat_equation):
```math
m \dfrac{d u}{d t} = \alpha \left( \dfrac{d^2 u}{d x^2} + \dfrac{d^2 u}{d y^2} \right) \, .
```
Here, no source term $\sigma$ is present. The neural network learns to solve a rectangular plate that is exposed to heat at one vertex.


## Raissi's Burgers Equation (discrete-time) 
The **fifth** example is Raissi's discrete-time solution to [Burgers' Equation](https://en.wikipedia.org/wiki/Burgers%27_equation):
This solves the PDE with only two time-snapshots at t0 and t1. The neural network outputs $q$ stages of the [Runge-Kutta](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods) method, representing the time discretization. Raissi's paper reports an error of 0.007 [PINN Part I](https://arxiv.org/pdf/1711.10561.pdf). Here 0.0051 is achieved. Obviously, the training did not fully converge. The boundary conditions are not met.  

<img src="https://github.com/JanAlexanderZak/pinn_examples/blob/main/src/discrete_time/raissi_allen_cahn/pred_t1.png" width="400" height="250" />

The training exhibits a high volatility. Three examplatory models are shown in the following figure. Other versions were significantly worse or did not start convergence at all.

<img src="https://github.com/JanAlexanderZak/pinn_examples/blob/main/src/discrete_time/raissi_allen_cahn/loss_plot.png" width="400" height="250" />

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

**Prateek Bhustali** (TU Delft)  
- https://github.com/omniscientoctopus/Physics-Informed-Neural-Networks

**Ian Henderson** (University of Toulouse)  
- https://towardsdatascience.com/physics-informed-neural-networks-pinns-an-intuitive-guide-fff138069563

**Daniel Crews**  
- https://github.com/crewsdw/pinns_project

**Jay Roxis**  
- https://github.com/jayroxis/PINNs
