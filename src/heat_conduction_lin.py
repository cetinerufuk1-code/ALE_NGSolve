"""" Head conduction equations Arbitrary Eularian Lagrangian (ALE) form.

    Testing linearized mass term solution and comparing them to the standard
    linear solve.

    STATUS: In progress."""

import ngsolve as ngs
import numpy as np
import netgen.gui # do not remove this
from netgen.occ import Glue
from netgen.occ import MoveTo
from netgen.occ import OCCGeometry
from netgen.occ import Rectangle
from netgen.occ import X
from netgen.occ import Y
from ngsolve.internal import visoptions

# ================ PARAMETERS =================
ORDER = 3
MAX_ELEMENT_SIZE = 0.05
TIME_STEP = 0.001
END_TIME = 0.5
SOURCE_STRENGTH = 100
EPS = 1e-5  # delta for finite difference


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
    _, deformation_jacobian, inverse_deformation_gradient = (
    grid_deformation_quantities(mesh, grid_deformation)
    )

    # Analytical Derivative Calculation
    deformation_determinant_derivative = (ngs.Trace(
        deformation_jacobian * inverse_deformation_gradient *
        ngs.Grad(delta_grid),
    ))

    return (deformation_determinant_derivative *
        ngs.InnerProduct(temperature_trial, temperature_test) / TIME_STEP
        ) * ngs.dx

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
    
    # Define the heat conduction equations in weak form
    lhs = ngs.BilinearForm(temperature_space, symmetric=False)
    rhs = ngs.LinearForm(temperature_space)

    lhs += mass_integral(mesh, grid_deformation, temperature_trial, temperature_test)
    lhs += diffusion_integral(mesh, grid_deformation, temperature_trial, temperature_test)
    lhs += convection_integral(mesh, grid_deformation,
                            grid_deformation_old, temperature_trial, temperature_test)

    rhs += source_integral(mesh, grid_deformation, temperature_test)
    rhs += mass_integral(mesh, grid_deformation, temperature_old, temperature_test)

    # LHS & RHS for linearized Mass term solve
    grid_steady = ngs.GridFunction(displacement_space)
    grid_steady.Set(ngs.CF((0,0)))
    delta_grid_steady = grid_deformation - grid_steady

    lhs_steady = ngs.BilinearForm(temperature_space, symmetric=False)
    rhs_steady = ngs.LinearForm(temperature_space)

    # Linearized mass term
    lhs_steady += (
        mass_integral(mesh,grid_steady,temperature_trial_lin,temperature_test_lin) +
        mass_integral_wrt_deformation(mesh,grid_steady,temperature_trial,temperature_test,delta_grid_steady)
    )
    lhs_steady += diffusion_integral(mesh, grid_deformation, temperature_trial_lin, temperature_test_lin)
    lhs_steady += convection_integral(mesh, grid_deformation,
                            grid_deformation_old, temperature_trial_lin, temperature_test_lin)

    rhs_steady += (
        mass_integral(mesh,grid_steady,temperature_old_lin,temperature_test_lin) +
        mass_integral_wrt_deformation(mesh,grid_steady,temperature_old_lin,temperature_test,delta_grid_steady)
    )
    rhs_steady += source_integral(mesh, grid_deformation, temperature_test_lin)
   
    preconditioner = ngs.Preconditioner(lhs, type="multigrid", inverse="sparsecholesky")
    preconditioner_steady = ngs.Preconditioner(lhs_steady, type="multigrid", inverse="sparsecholesky")

    # Initial Conditions (this might be unnecessary)
    temperature.Set(0.0)  
    temperature_lin.Set(0.0)

    # Begin visualization
    ngs.Draw(temperature, mesh, "temperature")
    ngs.Draw(grid_deformation, mesh, "displacement")
    visoptions.scalfunction = "temperature:0"
    visoptions.vecfunction = "displacement"
    visoptions.deformation = 1

    # Begin Time Iteration
    t = 0
    with ngs.TaskManager():
        while t < END_TIME:
            temperature_old.vec.data = temperature.vec
            temperature_old_lin.vec.data = temperature_lin.vec
            grid_deformation_old.vec.data = grid_deformation.vec

            # Assign some deformation
            displace_x = (0.035 * ngs.sin(ngs.pi * ngs.x / 0.5 / 2) *
                ngs.sin(ngs.pi * ngs.y / 0.5) * ngs.sin((10 * ngs.pi) * t))
            displace_y = (0.035 * ngs.sin(ngs.pi * ngs.y / 0.5 / 2) *
                ngs.sin(ngs.pi * ngs.x / 0.5) * ngs.sin((10 * ngs.pi) * t))

            grid_deformation.Set(
                ngs.CF((displace_x, displace_y)),
            )

            # Solve
            standard_solver(lhs,rhs,preconditioner,temperature)

            # Display Current timestep
            ngs.Redraw(blocking=True)

            # Calculate Mass Derivative
            # To-Do: Implement function for each derivative
            delta_grid = grid_deformation - grid_deformation_old
            mass_derivative_analytical = ngs.BilinearForm(temperature_space, symmetric=False)
            mass_derivative_analytical += (mass_integral_wrt_deformation(
                mesh, grid_deformation, temperature_trial, temperature_test, delta_grid)
            )
            mass_derivative_matrix_analytical = mass_derivative_analytical.Assemble().mat

            # Mass Derivative Finite Difference
            grid_deformation_dd = ngs.GridFunction(displacement_space)
            grid_deformation_dd.Set(grid_deformation + EPS * delta_grid)

            mass_derivative_finite = ngs.BilinearForm(temperature_space, symmetric=False)
            mass_derivative_finite += (1 / EPS) * (
                mass_integral(mesh, grid_deformation_dd, temperature_trial, temperature_test) -
                mass_integral(mesh, grid_deformation, temperature_trial, temperature_test))

            mass_derivative_matrix_finite = mass_derivative_finite.Assemble().mat

            # Validate analytical matrix values
            _, _, va = mass_derivative_matrix_analytical.COO()
            _, _, vb = mass_derivative_matrix_finite.COO()

            np.testing.assert_allclose(va, vb, rtol=0, atol=1e-8, strict=False)

            # Linearized mass term solve
            standard_solver(lhs_steady,rhs_steady,preconditioner_steady,temperature_lin)

            # Report the difference
            error = np.sqrt(ngs.Integrate(
                (temperature_lin - temperature)**2,
                mesh
            ))      

            print(f"Current absolute error : {error:.3e}")

            t += TIME_STEP

    input("Press any key to exit...")


if __name__ == "__main__":
    main()
