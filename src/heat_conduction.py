"""" Head conduction equations Arbitrary Eularian Lagrangian (ALE) form."""

import ngsolve as ngs
import numpy as np
from netgen.meshing import MeshingParameters
from netgen.occ import Rectangle, Glue, X, Y, MoveTo
from netgen.occ import OCCGeometry
import netgen.gui
from time import sleep
from ngsolve.internal import visoptions
import matplotlib.pyplot as plt

print(dir(netgen.gui))
# ================ PARAMETERS =================
ORDER = 3
MAX_ELEMENT_SIZE = 0.05
TIME_STEP = 0.001
END_TIME = 0.5
SOURCE_STRENGTH = 100

def generate_mesh(
        order: int,
        max_element_size: float
    ) -> ngs.Mesh:
    """Generates the mesh for the ciruclar cylinder to be used in the simulation.

    Args:
        order: Order of the mesh cells.
        max_element_size: Globabl maximum mesh size. Does not effect,
        mesh size near the cylinder.

    Returns:
        Mesh: The resulting mesh.
    """
    shell = Rectangle(0.5,0.5).Face()
    shell.faces.name = "solid"
    shell.edges.Min(X).name  = "left"
    shell.edges.Max(X).name = "right"
    shell.edges.Min(Y).name = "down"
    shell.edges.Max(Y).name = "up"

    hot = MoveTo(0.25-(0.2*0.5),0.25-(0.2*0.5)).Rectangle(0.2,0.2).Face()
    hot.faces.name = "hot"

    domain = shell - hot

    geo = Glue([domain,hot])

    mesh = ngs.Mesh(OCCGeometry(geo,dim = 2).GenerateMesh(maxh=max_element_size))
    mesh.Curve(order)

    return mesh

def mesh_deformation_quantities(
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
        jacobian:
        inverse_deformation_gradient:
    """
    identity = ngs.Id(mesh.dim)
    deformation_gradient = ngs.Grad(grid_deformation) + identity
    jacobian = ngs.Det(deformation_gradient)
    inverse_deformation_gradient = ngs.Inv(deformation_gradient)

    return (
        deformation_gradient,
        jacobian,
        inverse_deformation_gradient,
    )

def get_mass_wrt_deformation(
        mesh : ngs.Mesh,
        grid_deformation : ngs.GridFunction,
        temperature: ngs.GridFunction,
        test_function : ngs.GridFunction,
        delta_grid : ngs.CoefficientFunction,
        printerr : bool = False
    ) -> tuple[
        float,
        float]:
    """Calculates derivative of mass term wrt to the grid deformation.

        Calculates analytical derivative and the finite difference derivative. Returns both values
        and the difference if enabled.

        Args: 
            mesh: current finite element mesh.
            grid_deformation: grid deformation at the current timestep.
            temperature: Solution field at the current timestep.
            test_function: Test function to evaluate the mass term integral.
            delta_grid: Change in grid deformation.
            printerr: Turn on/off the current error reprot output.
        
        Returns:
            mass_wrt_deformation_analytical.
            mass_wrt_deformation_finite.
    """
    epsillon = 1e-3 # delta for the finite difference

    finite_space = grid_deformation.space
    grid_deformation_dd = ngs.GridFunction(finite_space)
    grid_deformation_dd.Set(grid_deformation + epsillon * delta_grid)
    _, deformation_jacobian, inverse_deformation_gradient  = mesh_deformation_quantities(mesh,grid_deformation)
    
    _, deformation_jacbian_dd, _ = mesh_deformation_quantities(mesh,grid_deformation_dd)

    # Analytical Derivative Calculation
    deformation_jacobian_derivative = (ngs.Trace(
        deformation_jacobian * inverse_deformation_gradient *
        ngs.Grad(delta_grid)
    ))

    mass_wrt_deformation_analytical = ngs.Integrate(deformation_jacobian_derivative *
            temperature * test_function / TIME_STEP, mesh
    )

    # Finite Differene Derivative Calculation
    mass_wrt_deformation_finite = (
        ( ngs.Integrate(deformation_jacbian_dd * temperature * test_function / TIME_STEP, mesh) -
        ngs.Integrate(deformation_jacobian * temperature * test_function / TIME_STEP, mesh) ) /
        epsillon
    )

    if printerr:
        absErr = np.abs(mass_wrt_deformation_finite - mass_wrt_deformation_analytical)
        print(f"Current mass derivative difference: {absErr:.3e}")

    return (mass_wrt_deformation_analytical, mass_wrt_deformation_finite)

def plot_results(
    mass_wrt_deformation_history : list[tuple[float,float]],
    time_history : list[float]
    ) -> None:
    """Plots calculated derivatives.

        Args:
            mass_wrt_deformation_hisotry: list of tuples containing the history
            of both analytical and finite difference solutions.
            time_history: List of time values corresponding to the derivative values above.
    """

    mass_wrt_deformation_analytical = [item[0] for item in mass_wrt_deformation_history]
    mass_wrt_deformation_finite = [item[1] for item in mass_wrt_deformation_history]
    
    fig, ax1 = plt.subplots()

    ax1.plot(time_history,mass_wrt_deformation_finite,label="Finite Difference Approximation",linewidth=3)
    ax1.plot(time_history,mass_wrt_deformation_analytical,linestyle="--",label="Analytical Approximation",linewidth=3)

    ax1.legend()
    ax1.set_ylabel(r"$\frac{\partial M}{\partial d}\,\delta d$")
    ax1.set_xlabel("Time (s)")
    plt.show()


def main() -> None:
    """"Main setup and time iteration."""
    mesh = generate_mesh(ORDER,MAX_ELEMENT_SIZE)
    
    # Setup Displacement and Temperature Fields
    displacement_space = ngs.VectorH1(mesh, order=ORDER)
    grid_deformation = ngs.GridFunction(displacement_space)
    grid_deformation_old = ngs.GridFunction(displacement_space)

    temperature_space = ngs.H1(mesh,order = ORDER,dirichlet ="left|right|up|down")
    (temperature_trial) , (temperature_test) = temperature_space.TnT()
    temperature, temperature_old = (
        ngs.GridFunction(temperature_space), ngs.GridFunction(temperature_space))
    
    # Calculate mesh deformation quanitites
    _, deformation_jacobian, inverse_deformation_gradient = (
        mesh_deformation_quantities(mesh,grid_deformation)
    )
    # Define heat conduction weak form
    true_compile = False
    lhs = ngs.BilinearForm(temperature_space, symmetric=False)
    rhs = ngs.LinearForm(temperature_space)

    # Mass
    mass = (deformation_jacobian * ngs.InnerProduct(
        (temperature_trial / TIME_STEP) , temperature_test)
    ) * ngs.dx

    diffusion = (deformation_jacobian * ngs.InnerProduct(
        inverse_deformation_gradient.trans * ngs.Grad(temperature_trial),
        inverse_deformation_gradient.trans * ngs.Grad(temperature_test))
    ) * ngs.dx

    conv = -(deformation_jacobian * ngs.InnerProduct(
        inverse_deformation_gradient.trans * ngs.Grad(temperature_trial),
        (grid_deformation - grid_deformation_old) / TIME_STEP * temperature_test)
    ) * ngs.dx

    source = (deformation_jacobian * 
              SOURCE_STRENGTH * temperature_test
    ) * ngs.dx("hot")

    transient = (deformation_jacobian * ngs.InnerProduct(
        (temperature_old / TIME_STEP) , temperature_test)
    ) * ngs.dx

    # Define LHS, RHS
    lhs += (mass + diffusion + conv)
    rhs += (transient + source)
    c = ngs.Preconditioner(lhs, type="multigrid", inverse="sparsecholesky")

    # Initial Conditions 
    temperature.Set(0.0)

    # Begin Time Iteration
    t = 0
    test_function = ngs.GridFunction(temperature_space)
    test_function.Set(1.0)

    ngs.Draw(temperature,mesh,"temperature")
    ngs.Draw(grid_deformation,mesh,"displacement")

    visoptions.scalfunction = "temperature:0"
    visoptions.vecfunction = "displacement"
    visoptions.deformation = 1

    mass_wrt_deformation_history : list[tuple[float,float]] = []
    time_history : list[float] = []
    
    with ngs.TaskManager():
        while t < END_TIME:
            temperature_old.vec.data = temperature.vec
            grid_deformation_old.vec.data = grid_deformation.vec

            # Assign some deformation
            displace_x = 0.035 * ngs.sin(ngs.pi * ngs.x / 0.5 / 2) * ngs.sin(ngs.pi * ngs.y / 0.5) * ngs.sin((10*ngs.pi)*t)
            displace_y = 0.035 * ngs.sin(ngs.pi * ngs.y / 0.5 / 2) * ngs.sin(ngs.pi * ngs.x / 0.5) * ngs.sin((10*ngs.pi)*t)

            grid_deformation.Set(
                ngs.CF((displace_x,displace_y))
            ) 
            
            # Solve 
            lhs.Assemble()
            rhs.Assemble()
            inv = ngs.CGSolver(lhs.mat,c.mat,printrates=False)
            temperature.vec.data = (
                inv * rhs.vec)

            # Display Current timestep
            # (deformation set for visualization only)
            ngs.Redraw(blocking=True)

            delta_grid = grid_deformation - grid_deformation_old
            # Calculate Derivatives
            mass_wrt_deformation = get_mass_wrt_deformation( 
                mesh,grid_deformation,temperature,test_function,delta_grid,True)

            mass_wrt_deformation_history.append(mass_wrt_deformation)
            time_history.append(t)

            t += TIME_STEP 

    # Plot Results
    plot_results(mass_wrt_deformation_history,time_history)
    input("Press any key to exit...")


if __name__ == "__main__":
    main()
