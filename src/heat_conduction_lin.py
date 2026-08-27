"""" Head conduction equations Arbitrary Eularian Lagrangian (ALE) form.

    Platform for testing and validating the steady-state linearized Arbitrary 
    Lagrangian–Eulerian (ALE) heat conduction equations by comparing against
    standard non-linear heat conduction formulation. The solver provides access 
    to derivatives of the individual terms, enabling detailed verification, 
    analysis, and comparison of the linearized and non-linear formulations.

    STATUS: In progress."""

import argparse
import ngsolve as ngs
import numpy as np
from netgen.occ import Glue
from netgen.occ import MoveTo
from netgen.occ import OCCGeometry
from netgen.occ import Rectangle
from netgen.occ import X
from netgen.occ import Y
from ngsolve.internal import visoptions

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments and set defaults for simulation parameters.
    
    Returns:
        vis_option."""
    parser = argparse.ArgumentParser(
        description="Run simulation.",
    )
    parser.add_argument(
        "--vis-on",
        action="store_true",
        help="Turn on/off visualization, by default set to False.",
    )
    return parser.parse_args()

# ================ PARAMETERS =================
args = parse_arguments() 

ORDER = 3
MAX_ELEMENT_SIZE = 0.05
TIME_STEP = 0.001
END_TIME = 0.5
SOURCE_STRENGTH = 100
VIS_OPTION = args.vis_on
MAX_DIFF = 1e-2 # enforce less than 1% difference
AMPLITUDE = 1e-2

if VIS_OPTION: # check if user wants visualization
    import netgen.gui

def generate_mesh(
        order: int,
        max_element_size: float,
    ) -> ngs.Mesh:
    """Generates the mesh for the rectangular domain for the heat conduction problem.

        Features a rectangular section in the middle labeled "hot" to
        assign for the source term.

    Args:
        order: Order of the mesh cells.
        max_element_size: Globabl maximum mesh size. Does not effect,
        mesh size near the cylinder.

    Returns:
        Mesh: The resulting mesh.
    """
    shell = Rectangle(0.5, 0.5).Face()
    shell.faces.name = "solid"
    shell.edges.Min(X).name = "left"
    shell.edges.Max(X).name = "right"
    shell.edges.Min(Y).name = "down"
    shell.edges.Max(Y).name = "up"

    hot = MoveTo(0.25 - (0.2 * 0.5), 0.25 - (0.2 * 0.5)).Rectangle(0.2, 0.2).Face()
    hot.faces.name = "hot"

    domain = shell - hot

    geo = Glue([domain, hot])

    mesh = ngs.Mesh(OCCGeometry(geo, dim=2).GenerateMesh(maxh=max_element_size))
    mesh.Curve(order)

    return mesh

def grid_deformation_quantities(
        mesh: ngs.Mesh,
        grid_deformation: ngs.GridFunction,
    ) -> tuple[
        ngs.CoefficientFunction,
        ngs.CoefficientFunction,
        ngs.CoefficientFunction,
    ]:
    """Calculates and returns necessary grid deformation quantites.

    Args:
        mesh: NGSolve mesh used in the current setup.
        grid_deformation: Mesh displacement field given as
        NGSolve grid function.

    Returns:
        deformation_gradient:
        determinant:
        inverse_deformation_gradient:
    """
    identity = ngs.Id(mesh.dim)
    deformation_gradient = ngs.Grad(grid_deformation) + identity
    determinant = ngs.Det(deformation_gradient)
    inverse_deformation_gradient = ngs.Inv(deformation_gradient)

    return (
        deformation_gradient,
        determinant,
        inverse_deformation_gradient,
    )


def mass_integral(
        mesh: ngs.Mesh,
        grid_deformation: ngs.GridFunction,
        temperature_trial: ngs.CoefficientFunction,
        temperature_test: ngs.CoefficientFunction,
    ) -> ngs.comp.SumOfIntegrals:
    """ALE heat conduction, mass term.

    Symbolic Integral: ∫ J⋅(u/Δt)⋅v dΩ₀

    Args:
        mesh: Current finite element mesh.
        grid_deformation: Deformation '𝚽'.
        temperature_trial: Trial function.
        temperature_test: Test function.

    Returns:
        mass_term.
    """
    _, deformation_determinant, _ = (
        grid_deformation_quantities(mesh, grid_deformation))

    return (deformation_determinant * ngs.InnerProduct(
        temperature_trial, temperature_test) / TIME_STEP
        ) * ngs.dx


def diffusion_integral(
    mesh: ngs.Mesh,
    grid_deformation: ngs.GridFunction,
    temperature_trial: ngs.CoefficientFunction,
    temperature_test: ngs.CoefficientFunction,
) -> ngs.comp.SumOfIntegrals:
    """ALE heat conduction, diffusion term.

    Symbolic integral: ∫ J⋅(F⁻ᵀ∇T)⋅(F⁻ᵀ∇v) dΩ₀

    Args:
        mesh: Current finite element mesh.
        grid_deformation: Deformation '𝚽'.
        temperature_trial: Trial function.
        temperature_test: Test function.

    Returns:
        diffusion_term.
    """
    _, deformation_determinant, inverse_deformation_gradient = (
        grid_deformation_quantities(mesh, grid_deformation)
    )

    return (deformation_determinant * ngs.InnerProduct(
        (inverse_deformation_gradient.trans * ngs.Grad(temperature_trial)),
        (inverse_deformation_gradient.trans * ngs.Grad(temperature_test)))
        ) * ngs.dx


def convection_integral(
    mesh: ngs.Mesh,
    grid_deformation: ngs.GridFunction,
    grid_deformation_old: ngs.GridFunction,
    temperature_trial: ngs.CoefficientFunction,
    temperature_test: ngs.CoefficientFunction,
) -> ngs.comp.SumOfIntegrals:
    """ALE heat conduction, convection term.

    Symbolic integral: ∫ -JF⁻ᵀ⋅∇u⋅d𝛷/dt⋅v dΩ₀

    Args:
        mesh: Current finite element mesh.
        grid_deformation: Deformation '𝚽'.
        grid_deformation_old : Deformation at previous timestep.
        temperature_trial: Trial function.
        temperature_test: Test function.

    Returns:
        convection_term.
    """
    _, deformation_determinant, inverse_deformation_gradient = (
        grid_deformation_quantities(mesh, grid_deformation)
    )

    return -(deformation_determinant * ngs.InnerProduct(
        inverse_deformation_gradient.trans * ngs.Grad(temperature_trial),
        (grid_deformation - grid_deformation_old) / TIME_STEP * temperature_test)
        ) * ngs.dx


def source_integral(
    mesh: ngs.Mesh,
    grid_deformation: ngs.GridFunction,
    temperature_test: ngs.CoefficientFunction,
) -> ngs.comp.SumOfIntegrals:
    """ALE heat conduction, source term.

    Symbolic integral: ∫ J⋅f⋅v dΩ_hot

    Args:
        mesh: Current finite element mesh.
        grid_deformation: Deformation '𝚽'.
        temperature_test: Test function.

    Returns:
        source_term.
    """
    _, deformation_determinant, _ = (
    grid_deformation_quantities(mesh, grid_deformation)
    )

    return (deformation_determinant * SOURCE_STRENGTH *
        temperature_test) * ngs.dx("hot")


def mass_integral_wrt_deformation(
        mesh: ngs.Mesh,
        grid_deformation: ngs.GridFunction,
        temperature_trial: ngs.GridFunction,
        temperature_test: ngs.GridFunction,
        delta_grid: ngs.CoefficientFunction,
    ) -> ngs.comp.SumOfIntegrals:
    """Calculates derivative of mass term wrt to the grid deformation.

    Symbolic integral: ∂M/∂𝚽⋅δ𝚽 = ∫ J⋅tr(F⁻¹⋅∇δ𝚽)⋅u⋅v/Δt dΩ₀

    Args:
        mesh: current finite element mesh.
        grid_deformation: grid deformation at the current timestep.
        temperature_trial: Trial function
        temperature_test: Test function.
        delta_grid: Change in grid deformation.

    Returns:
        mass_integral_wrt_deformation.
    """
    _, deformation_determinant, inverse_deformation_gradient = (
    grid_deformation_quantities(mesh, grid_deformation)
    )

    # Analytical Derivative Calculation
    deformation_determinant_derivative = (ngs.Trace(
        deformation_determinant * inverse_deformation_gradient *
        ngs.Grad(delta_grid)))

    return (deformation_determinant_derivative *
        ngs.InnerProduct(temperature_trial, temperature_test) / TIME_STEP
        ) * ngs.dx()

def diffusion_integral_wrt_deformation(
    mesh: ngs.Mesh,
    grid_deformation: ngs.GridFunction,
    temperature_trial: ngs.GridFunction,
    temperature_test: ngs.GridFunction,
    delta_grid: ngs.CoefficientFunction,
) -> ngs.comp.SumOfIntegrals:       
    _, deformation_determinant, inverse_deformation_gradient = (
    grid_deformation_quantities(mesh, grid_deformation)
    )

    deformation_determinant_derivative = (ngs.Trace(
        deformation_determinant * inverse_deformation_gradient *
        ngs.Grad(delta_grid)))

    inverse_deformation_gradient_derivative = ( -1 *
        inverse_deformation_gradient * ngs.Grad(delta_grid) *
        inverse_deformation_gradient
    )

    return (
        deformation_determinant_derivative * ngs.InnerProduct(
            inverse_deformation_gradient.trans * ngs.Grad(temperature_trial),
            inverse_deformation_gradient.trans * ngs.Grad(temperature_test)
        ) * ngs.dx + 
        deformation_determinant * ngs.InnerProduct(
            inverse_deformation_gradient_derivative.trans * ngs.Grad(temperature_trial),
            inverse_deformation_gradient.trans * ngs.Grad(temperature_test)
        ) * ngs.dx + 
        deformation_determinant * ngs.InnerProduct(
            inverse_deformation_gradient * ngs.Grad(temperature_trial),
            inverse_deformation_gradient_derivative.trans * ngs.Grad(temperature_test)
        ) * ngs.dx
    )

def convection_integral_wrt_deformation(
    mesh: ngs.Mesh,
    grid_deformation: ngs.GridFunction,
    grid_deformation_old: ngs.GridFunction,
    temperature_trial: ngs.GridFunction,
    temperature_test: ngs.GridFunction,
    delta_grid: ngs.CoefficientFunction,
    delta_grid_old : ngs.CoefficientFunction
) -> ngs.comp.SumOfIntegrals:
    """Calculates derivative of convection term wrt to grid deformation
    
    Symbolic intgral: 
    
    Args:
        mesh: current finite element mesh.
        grid_deformation: grid deformation at the current timestep.
        grid_deformation_old : grid deformation at previous timestep.
        temperature_trial: Trial function
        temperature_test: Test function.
        delta_grid: Change in grid deformation at the current timestep.
        delta_grid_old: change in grid deformation at previous timestep.

    Returns:
        convection_integral_wrt_deformation.
    """
    _ , deformation_determinant, inverse_deformation_gradient = (
        grid_deformation_quantities(mesh,grid_deformation)
    )

    deformation_determinant_derivative = (ngs.Trace(
        deformation_determinant * inverse_deformation_gradient *
        ngs.Grad(delta_grid)))

    inverse_deformation_gradient_derivative = ( -1 *
        inverse_deformation_gradient * ngs.Grad(delta_grid) *
        inverse_deformation_gradient
    )

    derivative_determinant_inverse = (
        deformation_determinant_derivative * inverse_deformation_gradient.trans +
        deformation_determinant * inverse_deformation_gradient_derivative.trans
    )

    grid_deformation_derivative = (grid_deformation - grid_deformation_old) / TIME_STEP
    delta_grid_derivative= (delta_grid - delta_grid_old) / TIME_STEP

    return -1 * ( 
        ngs.InnerProduct(derivative_determinant_inverse * ngs.Grad(temperature_trial),
                         grid_deformation_derivative * temperature_test)
    ) * ngs.dx -(
        deformation_determinant *
        ngs.InnerProduct(inverse_deformation_gradient.trans * ngs.Grad(temperature_trial),
                         delta_grid_derivative * temperature_test)
    ) * ngs.dx



def source_integral_wrt_deformation(
    mesh: ngs.Mesh,
    grid_deformation: ngs.GridFunction,
    temperature_test: ngs.GridFunction,
    delta_grid: ngs.CoefficientFunction,
) -> ngs.comp.SumOfIntegrals:
    """Calculates derivative of source term wrt to grid deformation.
    
    Symbolic intgral: ∫ J⋅tr(F⁻¹⋅∇δ𝚽)⋅f⋅v dΩ₀
    Assuming source function is independent of grid deformation.
    
    Args:
        mesh: current finite element mesh.
        grid_deformation: grid deformation at the current timestep.
        temperature_test: Test function.
        delta_grid: Change in grid deformation.

    Returns:
        source_integral_wrt_deformation.
    """
    _, deformation_determinant, inverse_deformation_gradient = (
        grid_deformation_quantities(mesh, grid_deformation)
    )

    deformation_determinant_derivative = (ngs.Trace(
        deformation_determinant * inverse_deformation_gradient *
        ngs.Grad(delta_grid)))

    return (deformation_determinant_derivative *
            SOURCE_STRENGTH * temperature_test) * ngs.dx("hot")


def non_linear_form(
    mesh: ngs.Mesh,
    grid_deformation : ngs.GridFunction,
    grid_deformation_old : ngs.GridFunction,
    temperature_space : ngs.H1,
    temperature_trial : ngs.GridFunction,
    temperature_test : ngs.GridFunction,
    temperature_old : ngs.GridFunction
    ) -> tuple[
        ngs.BilinearForm,
        ngs.LinearForm
    ]:
    """LHS, RHS bilinear and linear form for the non-linear heat conduction eqautions.
    
    Args:
        mesh: current finite element mesh.
        grid_deformation: grid deformation at the current timestep.
        grid_deformation_old: grid deformation at previous timestep.
        temperature_space: current finite element space.
        temperature_trial: Trial function
        temperature_test: Test function.
        temperature_old : temperature field at previous timestep.
    
    Returns:
        lhs_nonlinear.
        rhs_nonlinear."""
    lhs_nonlinear = ngs.BilinearForm(temperature_space, symmetric=False)
    rhs_nonlinear = ngs.LinearForm(temperature_space)

    lhs_nonlinear += mass_integral(mesh, grid_deformation, temperature_trial, temperature_test)
    lhs_nonlinear += diffusion_integral(mesh, grid_deformation, temperature_trial, temperature_test)
    lhs_nonlinear += convection_integral(mesh, grid_deformation,
                            grid_deformation_old, temperature_trial, temperature_test)

    rhs_nonlinear += source_integral(mesh, grid_deformation, temperature_test)
    rhs_nonlinear += mass_integral(mesh, grid_deformation, temperature_old, temperature_test)

    return (lhs_nonlinear, rhs_nonlinear)

def steady_linearized_form(
    mesh: ngs.Mesh,
    grid_deformation : ngs.GridFunction,
    grid_deformation_old : ngs.GridFunction,
    grid_steady : ngs.GridFunction,
    delta_grid : ngs.GridFunction,
    delta_grid_old : ngs.GridFunction,
    temperature_space : ngs.H1,
    temperature_trial : ngs.GridFunction,
    temperature_test : ngs.GridFunction,
    temperature_old : ngs.GridFunction
    ) -> tuple[
        ngs.BilinearForm,
        ngs.LinearForm
    ]:        
    """LHS, RHS bilinear and linear form for the steady-state linearized heat conduction equations.
    
    Args:
        mesh: current finite element mesh.
        grid_deformation: grid deformation at the current timestep.
        grid_deformation_old: grid deformation at previous timestep.
        temperature_space: current finite element space.
        temperature_trial: Trial function
        temperature_test: Test function.
        temperature_old : temperature field at previous timestep.
    
    Returns:
        lhs_linear.
        rhs_linear.
    """
    lhs_linear = ngs.BilinearForm(temperature_space, symmetric =False)
    rhs_linear = ngs.LinearForm(temperature_space)
    # Mass 
    lhs_linear += (
        mass_integral(mesh,grid_steady,temperature_trial,temperature_test) +
        mass_integral_wrt_deformation(mesh,grid_steady,temperature_trial,temperature_test,delta_grid)
    )
    # Diffusion
    lhs_linear += (
        diffusion_integral(mesh, grid_steady, temperature_trial, temperature_test) +
        diffusion_integral_wrt_deformation(mesh,grid_steady,temperature_trial,temperature_test,delta_grid)
    )
    # Convection
    lhs_linear += convection_integral(mesh, grid_deformation,
                            grid_deformation_old, temperature_trial, temperature_test)

    # Transient
    rhs_linear += (
        mass_integral(mesh,grid_steady,temperature_old,temperature_test) +
        mass_integral_wrt_deformation(mesh,grid_steady,temperature_old,temperature_test,delta_grid)
    )
    # Source
    rhs_linear += (
        source_integral(mesh, grid_steady, temperature_test) +
        source_integral_wrt_deformation(mesh,grid_steady,temperature_test,delta_grid)
    )

    return (lhs_linear, rhs_linear)

def standard_solver(
        lhs : ngs.BilinearForm,
        rhs : ngs.BilinearForm,
        preconditioner : ngs.BilinearForm,
        temperature : ngs.GridFunction
) -> None:
    """Standard solver to compare with linearized results.
    
    Args:
        lhs: Bilinear form of the left hand side.
        rhs: Bilinear form of the right hand side.
        preconditioner: Preconditioner used in matrix inversion.
        temperature: Results to solve for."""
    lhs.Assemble()
    rhs.Assemble()
    inv = ngs.CGSolver(lhs.mat, preconditioner.mat, printrates=False)
    temperature.vec.data = (
        inv * rhs.vec)

def main() -> None:
    """"Main setup and time iteration."""
    mesh = generate_mesh(ORDER, MAX_ELEMENT_SIZE)

    # Setup Displacement and Temperature Fields
    displacement_space = ngs.VectorH1(mesh, order=ORDER)
    grid_deformation, grid_deformation_old = (
        ngs.GridFunction(displacement_space), ngs.GridFunction(displacement_space))

    temperature_space = ngs.H1(mesh, order=ORDER, dirichlet="left|right|up|down")
    (temperature_trial), (temperature_test) = temperature_space.TnT()
    temperature, temperature_old = (
        ngs.GridFunction(temperature_space), ngs.GridFunction(temperature_space))
    (temperature_trial_lin), (temperature_test_lin) = temperature_space.TnT()
    temperature_lin, temperature_old_lin = (
        ngs.GridFunction(temperature_space), ngs.GridFunction(temperature_space))
    
    # Define non-linear heat conduction weak form
    (lhs_nonlinear, rhs_nonlinear) = non_linear_form(
        mesh,grid_deformation,grid_deformation_old,
        temperature_space,temperature_trial,temperature_test,temperature_old)

    preconditioner = ngs.Preconditioner(lhs_nonlinear, type="multigrid", inverse="sparsecholesky")
        
    # LHS & RHS for linearized solve
    grid_steady = ngs.GridFunction(displacement_space)
    grid_steady.Set(ngs.CF((0,0))) # Assuming no displacement at equilibrium
    delta_grid_steady = grid_deformation - grid_steady
    delta_grid_steady_old = delta_grid_steady

    (lhs_linear, rhs_linear) = steady_linearized_form(
        mesh,grid_deformation,grid_deformation_old,grid_steady,delta_grid_steady,delta_grid_steady_old,
        temperature_space,temperature_trial_lin,temperature_test_lin,temperature_old_lin)
    
    preconditioner_steady = ngs.Preconditioner(lhs_linear, type="multigrid", inverse="sparsecholesky")

    if VIS_OPTION:
        netgen.gui.StartGUI
        ngs.Draw(temperature, mesh, "temperature")
        ngs.Draw(grid_deformation, mesh, "displacement")
        visoptions.scalfunction = "temperature:0"
        visoptions.vecfunction = "displacement"
        visoptions.deformation = 1

    # Begin Time Iteration
    t = 0
    non_lin_history = []

    with ngs.TaskManager():
        while t < END_TIME:
            temperature_old.vec.data = temperature.vec
            temperature_old_lin.vec.data = temperature_lin.vec
            grid_deformation_old.vec.data = grid_deformation.vec
            delta_grid_steady_old = delta_grid_steady

            # Assign some deformation
            displace_x = (AMPLITUDE * ngs.sin(ngs.pi * ngs.x / 0.5 / 2) *
                ngs.sin(ngs.pi * ngs.y / 0.5) * ngs.sin((10 * ngs.pi) * t))
            displace_y = (AMPLITUDE * ngs.sin(ngs.pi * ngs.y / 0.5 / 2) *
                ngs.sin(ngs.pi * ngs.x / 0.5) * ngs.sin((10 * ngs.pi) * t))

            grid_deformation.Set(
                ngs.CF((displace_x, displace_y)),
            )

            # Non-linear solve
            standard_solver(lhs_nonlinear,rhs_nonlinear,preconditioner,temperature)
            # Linearized solve
            standard_solver(lhs_linear,rhs_linear,preconditioner_steady,temperature_lin)

            # Calculate L2 Norm
            error = np.sqrt(ngs.Integrate(
                (temperature_lin - temperature)**2,
                mesh
            ))
            non_lin_history.append(
                np.sqrt(ngs.Integrate(
                    temperature**2,mesh
                )))
            normErr = error / np.max(non_lin_history)

            # Ensure difference is acceptable
            #assert (normErr <= MAX_DIFF), "Non-linear and linear solutions differ more than 10%."

            print(f"Time: {t:.2f}s - Current Normalized Error : {normErr:.3e}")
            t += TIME_STEP

            # Display Current timestep
            if VIS_OPTION:
                ngs.Redraw(blocking=True)


if __name__ == "__main__":
    main()
