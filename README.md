# Heat Conduction Arbitrary Lagrangian Eulerian (ALE) Form

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/6b05c2da-526b-4f6a-be5d-d7fa601d3998"
    alt="heat_conduction_example"
    width="500"
  />
</p>

This repository uses the NGSolve library to solve the heat conduction equations in an Arbitrary Lagrangian–Eulerian (ALE) formulation. Additional implementations linearize each contributing term and reformulate the solution field with respect to the steady-state solution.

## Theory

The weak form heat conduction equations in ALE form are given by the following:

$$\int_{\hat{\Omega}} J \left(\frac{\partial \hat{u}}{\partial t}\hat{v} - F^{-T}\nabla_{\hat{x}}\hat{u}\frac{\partial \Phi}{\partial t}\hat{v} + F^{-T}\nabla_{\hat{x}}\hat{u}F^{-T}\nabla_{\hat{x}}\hat{v} \right)\,d\hat{x} = \int_{\hat{\Omega}} J\hat{f}\hat{v}\,d\hat{x}$$

where F and J are a function of the grid deformation $\Phi$. 

$$F = (I + \nabla \Phi)$$
$$J =\mathrm{det}(F) = \mathrm{det}(I + \nabla \Phi)$$

The terms within the heat conduction equation are grouped and labeled as follows:

$$ \mathrm{Mass:} \int_{\hat{\Omega}}J\frac{\partial \hat{u}}{\partial t}\hat{v} = \int_{\hat{\Omega}}J\frac{u^{n+1}-u^{n}}{\Delta t}\hat{v}\,d\hat{x}$$

$$ \mathrm{Convection:} - \int_{\hat{\Omega}}JF^{-T}\nabla_{\hat{x}}\hat{u}\frac{\partial \Phi}{\partial t}\hat{v}\,d\hat{x}$$

$$ \mathrm{Diffusion:} \int_{\hat{\Omega}}JF^{-T}\nabla_{\hat{x}}\hat{u}F^{-T}\nabla_{\hat{x}}\hat{v}\,d\hat{x}$$

$$ \mathrm{Source:} \int_{\hat{\Omega}}J\hat{f}\hat{v}\,d\hat{x}$$

The mass term is discretized using the backward Euler method, with the solution at each time step obtained through a linear solve.

### Deriving Derivatives wrt to grid deformation

Each term in the heat conduction equation is derivated wrt to the grid deformation and the analytical form of each term is as follows:

$$ \mathrm{Mass:} \frac{\partial M}{\partial\phi} \delta\phi = \int_{\hat{\Omega}} (\frac{J}{\Delta t})\mathrm{tr}(F^{-1}\nabla(\delta\phi))\hat{u}\cdot\hat{v}\,d\hat{x}$$


(more derivatives to follow)
