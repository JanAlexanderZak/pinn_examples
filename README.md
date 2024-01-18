# PINN examples with a software-engineering approach

This repository is a collection of continuous-time and discrete-time physics-informed neural networks (PINNs) implemented in Pytorch Lightning.


The **first** example is Raissi's solution to [Burgers' Equation](https://en.wikipedia.org/wiki/Burgers%27_equation):
```math
\dfrac{d u}{d t} + u \dfrac{d u}{d x} = v \dfrac{d^2 u}{d x^2},
```
The aim is to train a neural network that can be used for inference. An implementation detail is that the time domain is added via a concatenated dataset. With increasing epochs from 500 to 20000, the solution gets approximated more accurately:  

![](https://github.com/JanAlexanderZak/pinn_examples/blob/main/src/continuous_time/raissi_burgers/raissi_burgers.gif)  

The **second** example is Moseley's identification the friction coefficient of the [damped harmonic oscillator](https://en.wikipedia.org/wiki/Harmonic_oscillator):
```math
m \dfrac{d^2 x}{d t^2} + \mu \dfrac{d x}{d t} + kx = 0,
```
An implementation detail is that the time domain is added as a hyperparameter. In this case, the evolution of $\mu$ is of interest. The plot is not reproduced here.  

![image](https://github.com/JanAlexanderZak/pinn_examples/blob/main/src/continuous_time/moseley_oscillator/mu_plot.png)  

The **third** example solves the one-dimensional [heat equation](https://en.wikipedia.org/wiki/Heat_equation):
```math
m \dfrac{d u}{d t} = \alpha \dfrac{d^2 u}{d x^2} + \sigma,
```
An implementation detail is the source term $\sigma$. The plot compares the analytical solution with the prediction of the PINN:  

![image](https://github.com/JanAlexanderZak/pinn_examples/blob/main/src/continuous_time/heat_eq_1d/analytical_vs_pinn.png)  


# Applied Ressources for continuous-time and discrete-time PINNs

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
