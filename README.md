# Heat Conduction Arbitrary Lagrangian Eulerian (ALE) Form

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/14cf01bf-2bd1-45f2-9ebf-a3559aecbc37"
    alt="heat_conduction_example_results"
    width="750"
  />
</p>

This repository uses the NGSolve library to solve the heat conduction equations in an Arbitrary Lagrangian–Eulerian (ALE) formulation. The main goal is to use the standard non-linear ALE form to verify steady-state linearized ALE solutions.

## Theory

The weak form heat conduction equations in ALE form are given by the following:

$$\int_{\hat{\Omega}} J \left(\frac{\partial \hat{u}}{\partial t}\hat{v} - F^{-T}\nabla_{\hat{x}}\hat{u}\frac{\partial \Phi}{\partial t}\hat{v} + F^{-T}\nabla_{\hat{x}}\hat{u}F^{-T}\nabla_{\hat{x}}\hat{v} \right)\,d\hat{x} = \int_{\hat{\Omega}} J\hat{f}\hat{v}\,d\hat{x}$$

where F and J are a function of the grid deformation $\Phi$. In this application the grid deformation is prescribed by a arbitrary function and therefore the deformation and the deformation quantities are known prior to the solve.

$$F = (I + \nabla \Phi)$$
$$J =\mathrm{det}(F) = \mathrm{det}(I + \nabla \Phi)$$

The terms within the heat conduction equation are grouped and labeled as follows:

$$ \mathrm{Mass:} \int_{\hat{\Omega}}J\frac{\partial \hat{u}}{\partial t}\hat{v}$$

$$ \mathrm{Convection:} - \int_{\hat{\Omega}}JF^{-T}\nabla_{\hat{x}}\hat{u}\frac{\partial \Phi}{\partial t}\hat{v}\,d\hat{x}$$

$$ \mathrm{Diffusion:} \int_{\hat{\Omega}}JF^{-T}\nabla_{\hat{x}}\hat{u}F^{-T}\nabla_{\hat{x}}\hat{v}\,d\hat{x}$$

$$ \mathrm{Source:} \int_{\hat{\Omega}}J\hat{f}\hat{v}\,d\hat{x}$$

The mass term is discretized with backward Euler:

$$\frac{\partial \hat u}{\partial t} \approx \frac{u^{n+1} - u^{n}}{\Delta t}$$

which splits the single mass integral into two pieces that end up on opposite sides of the linear system:

$$\int_{\hat{\Omega}}J\frac{\partial \hat{u}}{\partial t}\hat{v}d\hat{x} \approx \underbrace{\int_{\hat{\Omega}}J\frac{u^{n+1}}{\Delta t}\hat{v}d\hat{x}}_{\text{unknown } u^{n+1}\ \to\ \text{LHS}} \underbrace{\int_{\hat{\Omega}}J\frac{u^{n}}{\Delta t}\hat{v}d\hat{x}}_{\text{from previous step}\ \to\ \text{RHS}}$$

The first piece involves the trial function $\hat u = u^{n+1}$ and is assembled into the bilinear (LHS) form. The second piece uses $u^{n}$, which is already known from the previous timestep, and so is treated as data and moved to the linear (RHS) form. 

### A note on "non-linear" vs. "linear" terminology

The heat conduction PDE itself is linear in $\hat u$ in both formulations used in this repo. What varies between the two formulations is how the geometric quantities $J(\Phi)$ and $F^{-1}(\Phi)$ are treated:

- **"Non-linear form"** uses the exact $J(\Phi)$ and $F^{-1}(\Phi)$, evaluated directly from the current prescribed deformation $\Phi^{n+1}$, with no approximation. Because $\Phi$ is prescribed (not a function of $\hat u$), these coefficients are just known data at each timestep. Therfore even the "non-linear form" reduces to a single linear solve for $u^{n+1}$ per timestep. The name refers to the nonlinear dependence of $J$ and $F^{-1}$ on $\Phi$, not to any nonlinearity in the unknown temperature field.

- **"Linear(ized) form"** instead replaces $J(\Phi)$ and $F^{-1}(\Phi)$ with their first-order Taylor expansion in $\Phi$ about a fixed base/steady configuration $\Phi_0$ (see below).


### Steady-state linearization

The linearization is taken only with respect to the grid deformation $\Phi$, about a fixed reference/steady configuration $\Phi_0$.

$$\delta\Phi = \Phi - \Phi_0$$

Since $\Phi$ is prescribed at every timestep, $\delta\Phi$ is likewise a known quantity. Each term $T \in \{M, C, D, S\}$ is expanded to first order:

$$T(\Phi) \;\approx\; T(\Phi_0) \;+\; \left.\frac{\partial T}{\partial \phi}\right|_{\Phi_0} \delta\Phi$$

Applying this to every term gives the following:

$$\mathrm{LHS} \approx \Big[M(\Phi_0) + \tfrac{\partial M}{\partial\phi}\Big|_{\Phi_0}\!\delta\Phi\Big] + \Big[D(\Phi_0) + \tfrac{\partial D}{\partial\phi}\Big|_{\Phi_0}\!\delta\Phi\Big] + \Big[C(\Phi_0,\Phi_0) + \tfrac{\partial C}{\partial\phi}\Big|_{\Phi_0}\!\delta\Phi\Big]$$

$$\mathrm{RHS} \approx \Big[M(\Phi_0)\big|_{\hat u \to \hat u^n} + \tfrac{\partial M}{\partial\phi}\Big|_{\Phi_0}\!\delta\Phi\big|_{\hat u \to \hat u^n}\Big] + \Big[S(\Phi_0) + \tfrac{\partial S}{\partial\phi}\Big|_{\Phi_0}\!\delta\Phi\Big]$$

where $M$, $C$, $D$, $S$ are the mass, convection, diffusion, and source terms respectively. 

### Deriving derivatives with respect to grid deformation

Each term is differentiated with respect to $\phi$ in the direction $\delta\phi$. The two building-block identities, used repeatedly below, follow from standard results for the derivative of a determinant and an inverse:

$$\delta J = J\,\mathrm{tr}\!\left(F^{-1}\nabla(\delta\phi)\right)$$

$$\delta(F^{-T}) = -F^{-T}\,\nabla(\delta\phi)^{T}\,F^{-T}$$

**Mass:**

$$\frac{\partial M}{\partial\phi} \delta\phi = \int_{\hat{\Omega}} \left(\frac{J}{\Delta t}\right)\mathrm{tr}\!\left(F^{-1}\nabla(\delta\phi)\right)\hat{u}\cdot\hat{v}\,d\hat{x}$$

**Diffusion:**

$$\frac{\partial D}{\partial\phi}\delta\phi = \int_{\hat{\Omega}} \delta J \left(F^{-T}\nabla_{\hat{x}}\hat{u}\right)\cdot\left(F^{-T}\nabla_{\hat{x}}\hat{v}\right)\,d\hat{x}$$
$$+ \int_{\hat{\Omega}} J\left(\delta(F^{-T})\nabla_{\hat{x}}\hat{u}\right)\cdot\left(F^{-T}\nabla_{\hat{x}}\hat{v}\right)\,d\hat{x} + \int_{\hat{\Omega}} J\left(F^{-T}\nabla_{\hat{x}}\hat{u}\right)\cdot\left(\delta(F^{-T})\nabla_{\hat{x}}\hat{v}\right)\,d\hat{x}$$

**Convection:**

Writing $w = \partial\Phi/\partial t \approx (\Phi^{n+1}-\Phi^{n})/\Delta t$ and differentiating the product $JF^{-T}$ together with $w$:

$$\frac{\partial C}{\partial\phi}\delta\phi = -\int_{\hat{\Omega}} \Big[\delta J\, F^{-T} + J\,\delta(F^{-T})\Big]\nabla_{\hat{x}}\hat{u}\cdot w\,\hat{v}\,d\hat{x} - \int_{\hat{\Omega}} J F^{-T}\nabla_{\hat{x}}\hat{u}\cdot\delta w\,\hat{v}\,d\hat{x}$$

where $\delta w = (\delta\phi^{n+1} - \delta\phi^{n})/\Delta t$ is the discrete time-derivative of the deformation perturbation itself.

**Source:**

Since $\hat f$ is taken independent of $\Phi$, only $J$ contributes:

$$\frac{\partial S}{\partial\phi}\delta\phi = \int_{\hat{\Omega}_{\mathrm{hot}}} \delta J\,\hat{f}\,\hat{v}\,d\hat{x}$$


## Install Instructions

1. Install uv
```
pip install uv
```

2. Clone repository 
```
git clone <REPOSITOYRY_URL>
cd NGS_ALE_Heat
```

3. Create virtual environment and sync required packages
```
uv venv
uv sync
```

4. Run Script
```
uv run src\heat_conduction_lin.py --vis-on
```
