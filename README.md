# 2D Lid-Driven Cavity Flow solver using Lattice Boltzmann Method (LBM)

## Features
- D2Q9 Lattice
- Single Relaxation Time (SRT) or Bhatnagar-Gross-Krook (BGK) collision operator
- Velocity contours and streamlines
- Velocity profiles and validation against benchmark data (Ghia et al. (1982))

## Governing Equation

The Lattice Boltzmann Equation with BGK approximation

$\frac{\partial f}{\partial t} + \mathbf{c} \cdot \nabla f = \Omega(f) \Rightarrow f_i(\mathbf{x} + \mathbf{c}_i \Delta t, t + \Delta t) = f_i(\mathbf{x}, t) - \frac{1}{\tau} \left( f_i - f_i^{eq} \right)$


Equilibrium Distribution Function

$f_i^{eq} = w_i \rho \left[ 1 + \frac{\mathbf{c}_i \cdot \mathbf{u}}{c_s^2} + \frac{(\mathbf{c}_i \cdot \mathbf{u})^2}{2c_s^4} - \frac{\mathbf{u}^2}{2c_s^2} \right]$


## Macroscopic Quantities and Boundary Condition

Density:
$\rho = \sum_i f_i$


Velocity:
$\rho \mathbf{u} = \sum_i f_i \mathbf{c}_i$

Stationary Wall (Bounce-back BC):
$f_i = f_{opp}$

Moving Wall (Zou/He BC):
$f_i - f_i^{eq} = f_{opp} - f_{opp}^{eq}$
