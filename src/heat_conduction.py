"""" Head conduction equations Arbitrary Eularian Lagrangian (ALE) form."""

import ngsolve as ngs
import numpy as np
from netgen.meshing import MeshingParameters
from netgen.occ import Rectangle, Glue, X, Y, MoveTo
from netgen.occ import OCCGeometry
import netgen.gui
from time import sleep
from ngsolve.internal import visoptions

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
        mesh_displacement: ngs.GridFunction,
    ) -> tuple[
        ngs.CoefficientFunction,
        ngs.CoefficientFunction,
        ngs.CoefficientFunction,
    ]:
    """Calculates and returns necessary grid deformation quantites.

    Args:
        mesh: NGSolve mesh used in the current setup.
        mesh_displacement: Mesh displacement field given as
        NGSolve grid function.

    Returns:
        deformation_gradient:
        jacobian:
        inverse_deformation_gradient:
    """
    identity = ngs.Id(mesh.dim)
    deformation_gradient = ngs.Grad(mesh_displacement) + identity
    jacobian = ngs.Det(deformation_gradient)
    inverse_deformation_gradient = ngs.Inv(deformation_gradient)

    return (
        deformation_gradient,
        jacobian,
        inverse_deformation_gradient,
    )

def main() -> None:
    mesh = generate_mesh(ORDER,MAX_ELEMENT_SIZE)

    # ngs.Draw(mesh) ## Debug Line
    
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

    ngs.Draw(temperature,mesh,"temperature")
    ngs.Draw(grid_deformation,mesh,"displacement")

    visoptions.scalfunction = "temperature:0"
    visoptions.vecfunction = "displacement"
    visoptions.deformation = 1
    
    with ngs.TaskManager():
        while t < END_TIME:
            temperature_old.vec.data = temperature.vec
            grid_deformation_old.vec.data = grid_deformation.vec

            # Assign some deformation
            dist = ngs.sqrt(ngs.x*ngs.x + ngs.y * ngs.y)
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

            t += TIME_STEP 
            #sleep(0.05)
            # Slow down for better visualization
            
    input("Press any key to exit...")



if __name__ == "__main__":
    main()
