#=========================================================
#=========================================================
#   Solve LZ diff. equation with Neumann bc on 3D and 
#   Nische (wake) bc (DC-NM) on 1D complex network
#=========================================================
#=========================================================

"""
We impose D-bc on portion of 1D domain through Nitsche method.
We solve the problem with Conjugate Gradient method preconditioned with "metric AMG" method that
uses block Schwarz smoothers.
"""
import time
import sys
start = time.time()


from scipy.sparse import csr_matrix
from xii.assembler.average_matrix import average_matrix as average_3d1d_matrix, trace_3d1d_matrix
from block.algebraic.hazmath import block_mat_to_block_dCSRmat
from dolfin import *
from xii import *
import haznics
import time
from scipy.sparse import csr_matrix, save_npz, load_npz
from petsc4py import PETSc


from xii.assembler.ufl_utils import *
from xii.linalg.matrix_utils import is_number
'''
from ufl.corealg.traversal import traverse_unique_terminals
import dolfin as df
import ufl

'''



#parametri: termine reazione nel 3D: si 
#           termine reazione nell 1D: no
#           3D-neumann 1D-dirichelt
#           beta Nische Boundary
              

#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
def save_CSR(mat, path):
    if isinstance(mat, Vector):
        mat_np = np.array(mat)
    else:    
        mat_np  = ii_convert(mat).array() #<--numpy
    mat_csr = csr_matrix(mat_np) #<-- csr
    save_npz(f"{path}", mat_csr) #<-- saving





def get_mesh(n, coupling_radius, path_to_1Dmesh):

    '''
    mesh is from -1 to 1; with small tolerance for 1D radius 
    ADD INF AND MAX POINT AS INPUT
    '''

    #--------------------MESHES-------------------------

    #3D mesh--------------------------------------------
    inf_point = Point(-1 - 1.1 * coupling_radius, -1 - 1.1*coupling_radius, -1 - 1.1*coupling_radius)
    max_point = Point(1 + 1.1 * coupling_radius, 1 + 1.1*coupling_radius, 1 + 1.1*coupling_radius)
    meshV     = BoxMesh(inf_point, max_point, n, n, n)
    
    #--------------------------------------------------

    #1D mesh
    
    #LOADING NETWORK XDMF 1D MESH AND ITS TAGS (each mesh function must be saved in different files)
    meshQ = Mesh()
    with XDMFFile(f'{path_to_1Dmesh}marked_mesh.xdmf') as infile:
        infile.read(meshQ)

    Q_markers = MeshFunction('size_t', meshQ, 0)
    xdmf_file = XDMFFile(f'{path_to_1Dmesh}markers.xdmf') 
    
    xdmf_file.read(Q_markers)
    xdmf_file.close()
    
    #---------------------------------------------------------------
        
    #marking

    tag = 111# per input. 999 per output.

    ds = Measure('ds', domain=meshQ)
    ds = ds(subdomain_data=Q_markers)

    return meshV, meshQ, ds, tag
 




def get_system(meshV, meshQ, ds, tag, k3=1e-3, k1=1e-3, gamma=1e0, coupling_radius=0.):
    """A, b, W, bcs"""

   
    # Spaces
    V = FunctionSpace(meshV, 'CG', 1)
    
    # Access the dofmap and get the number of dofs
    dofmap = V.dofmap()
    V_DOF = dofmap.global_dimension()
    print("ecco 3D dofs", V_DOF)
  
    Q = FunctionSpace(meshQ, 'CG', 1)
    W = [V, Q]

    u, p = map(TrialFunction, W)
    v, q = map(TestFunction, W)
    '''---------------------------------------'''
       
      
    '''---------------------------------------'''
    #computing average operator------------------------------------------
   
    # Average (coupling_radius > 0) or trace (coupling_radius = 0)
    if coupling_radius > 0:
        # Averaging surface
        cylinder = Circle(radius=coupling_radius, degree=10)
        Ru, Rv   = Average(u, meshQ, cylinder), Average(v, meshQ, cylinder)
        C        = average_3d1d_matrix(V, Q, cylinder)
    else:
        Ru, Rv   = Average(u, meshQ, None), Average(v, meshQ, None)
        C        = trace_3d1d_matrix(V, Q, meshQ)

    #--------------------------------------------------------------------         
        
    # Line integral
    dx_ = Measure('dx', domain=meshQ)
   
    # Parameters
    k3, k1, gamma = map(Constant, (k3, k1, gamma))
    # f3, f1 = Expression('x[0] + x[1]', degree = 1), Constant(1)
    # f1     = Constant(1)


    # Set Nitsche Dirichlet boundary integral---------------------------
    '''
    Haznics preconditioner implementation does not support strong bc. 
    BC can be imposed strongly if Haznics is not used.
    '''

    #Nitsche parameters:
    h_E        = MaxCellEdgeLength(meshQ)
    n          = FacetNormal(meshQ)
    p_exact    = Constant(1) # 1D inlet boundary value
    beta_value = 5.
    beta       = Constant(beta_value)
    #-------------------------------------------------------------------

    # We're building a 2x2 problem with 3D reaction---------------------

    a = block_form(W, 2)

    a[0][0] = k3 * inner(grad(u), grad(v)) * dx + k3 * inner(u, v) * dx #3D Reaction

    a[1][1] = k1 * inner(grad(p), grad(q)) * dx  - inner(dot(grad(p), n), q) * ds(tag, domain=meshQ) - inner(p, dot(grad(q), n)) * ds(tag, domain=meshQ) +\
              beta*(h_E**-1) * inner(p, q) * ds(tag, domain=meshQ) 



    m = block_form(W, 2)
    m[0][0] = inner(Ru, Rv) * dx_
    m[0][1] = -inner(p, Rv) * dx_
    m[1][0] = -inner(q, Ru) * dx_
    m[1][1] = inner(p, q) * dx_

    L = block_form(W, 1)
    L[0] = inner(Constant(0),v) * dx
    L[1] = -inner(p_exact, dot(grad(q), n)) * ds(tag, domain=meshQ) +\
            beta*(h_E**-1) * inner(p_exact, q) * ds(tag, domain=meshQ)

    
    AD, M, b = map(ii_assemble, (a, m, L))

    # Coupling info
    C = csr_matrix(C.getValuesCSR()[::-1], shape=C.size)


    return (AD, M), b, W, C, V_DOF

#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------



def solve_haznics(W, A, b, AD, M, C):
    def block_to_haz(AA): 
        # first make sure the whole matrix is of block_mat type
        if hasattr(AA, 'block_collapse'):
            AA = AA.block_collapse()

        # then make sure each block is a petsc matrix
        brow, bcol = AA.blocks.shape
        for i in range(brow):
            for j in range(bcol):
                AA[i][j] = ii_collapse(AA[i][j])

        AAhaz = block_mat_to_block_dCSRmat(AA)

        return AAhaz

    dimW = sum([VV.dim() for VV in W])
    start_time = time.time()
    # convert vectors
    bb = ii_convert(b)
    b_np = bb[:]
    bhaz = haznics.create_dvector(b_np)
    xhaz = haznics.dvec_create_p(dimW)

    # convert matrices
    Ahaz = block_to_haz(A)
    Mhaz = block_to_haz(M)
    ADhaz = block_to_haz(AD)
    # coupling incidence matrix C
    csr0, csr1, csr2 = C.indptr, C.indices, C.data
    Chaz = haznics.create_matrix(csr2, csr1, csr0, C.shape[1])
    # print("\n------------------- Data conversion time: ", time.time() - start_time, "\n")

    # call solver
    niters = haznics.fenics_metric_amg_solver(Ahaz, bhaz, xhaz, ADhaz, Mhaz, Chaz)
     # Extract solution as numpy array
    haznics.dvec_write('/tmp/solution_raw.dat', xhaz)
    x_np = np.loadtxt('/tmp/solution_raw.dat', skiprows=1)
    return niters, xhaz, x_np





def getGraphDist(mesh1D, space):
    """
    input:

          mesh1D(dolfin mesh)
          mesh3D(dolfin mesh)

    output:
    
           dist(Mesh Function)
    """
    from closest_point_in_mesh import  closest_point_in_mesh
    '''Getting the  distance function from a mesh'''
    #import pdb; pdb.set_trace() 
    dist = Function(space)
    
    for i, cord in enumerate(space.tabulate_dof_coordinates()):
    
        close_p = closest_point_in_mesh(cord, mesh1D)
    
        dist.vector()[i] = np.linalg.norm(cord-close_p)

    return dist







#__________________________________________________________________________________________
#__________________________________________________________________________________________
#__________________________________________________________________________________________
#__________________________________________________________________________________________
#__________________________________________________________________________________________
#__________________________________________________________________________________________
#__________________________________________________________________________________________
#__________________________________________________________________________________________
#__________________________________________________________________________________________

#run the code

if __name__ == '__main__':
    import numpy as np

    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #parser.add_argument('-name', type=str, help='file name prefix')
    parser.add_argument('-mesh', type=int, help='file mesh prefix')
    parser.add_argument('-rad', type=float, help='vasculature radius')
    parser.add_argument('-nnn', type=int, help='mesh size')
    parser.add_argument('-which', type=str, help='train or test')
    args, _ = parser.parse_known_args()

    radius = args.rad  # radius (rho) of the averaging surface in (micro m??)
    #name = args.name 
    mesh_name = args.mesh
    which     = args.which
    # Parameters---------------------------------------------------------
    max_radius = 0.1
    sigma3d, sigma1d, kappa = 1e-3, 1, 1   

    #sigma3d, sigma1d, kappa = 1e-2, 1, 1     # ATTENTION!!! CHANGED CONSTANT 

    gamma = kappa * 2 * np.pi * radius  # coupling parameter
    sigma1d = sigma1d * np.pi * radius**2  # scaled 1d conductivity
    #mesh parameters-----------------------------------------------------
    nn = args.nnn

    #path_to_mats   = f"./FOM/mats/{name}_"
    path_to_1Dmesh = f'./nets/{which}/mynet__'

    #--------------------------------------------------------------------

    # Get discrete system
    meshV, meshQ, ds, tag   = get_mesh(nn, max_radius, path_to_1Dmesh)
    (AD, M), b, W, C, V_DOF = get_system(meshV, meshQ, ds, tag, k3=sigma3d, k1=sigma1d, gamma=gamma, coupling_radius=radius)
    A                       = AD + gamma * M  # gamma is the couling strength

   
    start_time = time.time()
    print("\n------------------ System setup and assembly time: ", time.time() - start_time, "\n")


    # Now solving Solve(CG + AMG)
    start        = time.time()
    niters, xhaz, x_np = solve_haznics(W, A, b, AD, M, C)
    end          = time.time()
    print("solving elapsed time", end - start)
    HAZ_time = end - start

      
    # Results
    dimV, dimQ = W[0].dim(), W[1].dim()
    print("************************")
    print("Parameters: ", f'{sigma3d=}, {sigma1d=}, {radius=}, {kappa=}', "\n")
    print(f'dim(V)={dimV} dim(Q)={dimQ}  hmax(V)={W[0].mesh().hmax():.2f}  hmin(V)={W[0].mesh().hmin():.2f}  hmin(Q)={W[1].mesh().hmin():.2f} '
          f'niters={niters}')
    print("************************")

    import os
    sol_folder = f'./solution/mesh{mesh_name}_rad{radius}_n{nn}'
    os.makedirs(sol_folder, exist_ok=True)

    # numpy save
   
    np.save(f'{sol_folder}/solution.npy', x_np)

    dimV, dimQ = W[0].dim(), W[1].dim()
    u3d = Function(W[0])
    u1d = Function(W[1])
    u3d.vector()[:] = x_np[:dimV]
    u1d.vector()[:] = x_np[dimV:]
 
    #import pdb; pdb.set_trace() 

    

    exit()

    "if you want you can continue and try to solve with standard AMG"

    #----------------------------------------------------------------------
    # Now solving with PETSc AMG
    AMGlev = 20
     
    b_PET = ii_convert(b)
    shape = b_PET.size()
    AD_PET = ii_convert(AD).mat()
    M_PET  = ii_convert(M).mat()
    A_PET  = AD_PET.axpy(gamma, M_PET) #<--total matrix AD + gamma * M
     
    b_PET = b_PET.vec()
    u     = np.zeros((shape))
    tol   = 1e-15
    u_PET = PETSc.Vec().createWithArray(u)
     
    # make solver
    # Create a KSP object and set the matrix
    ksp = PETSc.KSP().create()
    ksp.setOperators(A_PET)
    ksp.setType(PETSc.KSP.Type.GMRES)  # <--Choose the solver type
    ksp.setTolerances(rtol=tol)
    ksp.setPCSide(1)
    ksp.view()
    ksp.setFromOptions()  # <--Allow setting options from the command line or a file
     
    # Create and set up the preconditioner
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    pc.setHYPREType("boomeramg")
    # Access the options database
    opts = PETSc.Options()
     
    # Set the number of iterations                 # I GUESS OPTION MUST BE ENFORCED IN A DIFFEERENT WAY, OTHERWISE NO OTHER PETS OBJECT CAN BE USED LATED IN THE CODE 
    opts['pc_hypre_boomeramg_max_iter'] = 1
    # Set the maximum number of AMG levels
    opts['pc_hypre_boomeramg_max_levels'] = AMGlev
    # Set ILU type to none
    opts['pc_factor_mat_solver_package'] = 'none'
     
    # Call setFromOptions after setting options
    pc.setFromOptions()
     
    # Print preconditioner options
    pc.view()
     
     
    # SOLVE!
    ksp.solve(b_PET, u_PET)
     
    #----------------------------------------------------------------------




