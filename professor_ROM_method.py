# -----------------------------------------------------------------------------
#  Assembly & Solution script
#  --------------------------------------------------------------

import os
import warnings
warnings.filterwarnings('ignore')
import time
import argparse
from dolfin import *
import numpy as np
from scipy.sparse import coo_matrix
import pickle
from pathlib import Path
from lib.common_fun import *        # unchanged util imports
from lib.modules import *           # unchanged util imports
from scipy.spatial import cKDTree

import torch
from librom.packages import *
from dlroms import *
from dlroms.cores import GPU

from pdb import *


def has_mesh(m):
    return (m is not None) and (m.num_cells() > 0) and (m.num_vertices() > 0)


#### ROM SUPPORT FUNCTIONS

def loadROM(Vh, folder):
    seed = 10001 

    psi1 = PODMINN1(Vh, 9261, seed)
    psi1.load(f"./{folder}/psi1.npz")
    
    phi1_0, phi2_0, omega_0, decoder_0 = PODMINN2(Vh, len(V_u0), seed)
    phi1, phi2, omega, decoder = PODMINN2(Vh, len(V_u), seed)
    clos_0             = MINN_closure(Vh, seed)
    clos               = MINN_closure(Vh, seed)
    phi1_0.load(   f"./{folder}/phi1_0.npz")
    phi2_0.load(   f"./{folder}/phi2_0.npz")
    omega_0.load(  f"./{folder}/omega_0.npz")
    decoder_0.load(f"./{folder}/decoder_0.npz")
    clos_0.load(f"./{folder}/clos_0.npz")


    phi1.load(   f"./{folder}/phi1.npz")
    phi2.load(   f"./{folder}/phi2.npz")
    omega.load(  f"./{folder}/omega.npz")
    decoder.load(f"./{folder}/decoder.npz")
    clos.load(f"./{folder}/clos.npz")
    
    return psi1, phi1, phi2, omega, decoder, phi1_0, phi2_0, omega_0, decoder_0, clos_0, clos

def loadROM3(folder, kk):

    if kk == 0:
        coarsemap.load(f"./{folder}/coarsemap_0.npz")
    elif kk == 1:
        coarsemap.load(f"./{folder}/coarsemap_1.npz")
    else:
        coarsemap.load(f"./{folder}/coarsemap_k.npz")

    return coarsemap

def loadPOD(folder):

    V_u0 = GPU.tensor(np.load(f"./{folder}/V_u0.npy"))
    V_u = GPU.tensor(np.load(f"./{folder}/V_u.npy"))

    return V_u, V_u0

def loadbounds(folder):

    u3d_old_max = np.load(f"./{folder}/u3d_max.npy")
    u3d_old_min = np.load(f"./{folder}/u3d_min.npy")

    d_max = 0 #np.load(f"./{folder}/d_max.npy")
    d_min = 0 #np.load(f"./{folder}/d_min.npy")

    u1d_max = 0.07 #np.load(f"./ROM_DATA_step/ROM_DATA_step{step}/u1d_max.npy")
    u1d_min = 0.05 #np.load(f"./ROM_DATA_step/ROM_DATA_step{step}/u1d_min.npy")

    u3d_new_max = u3d_old_max
    u3d_new_min = u3d_old_min
    
    scaling_eta_ext = 5e13
    scaling_eta_dtn = 1e15
    scaling_eta_r0  = 5e11

    return u3d_old_min, u3d_old_max, u3d_new_min, u3d_new_max, d_min, d_max, u1d_min, u1d_max, scaling_eta_ext, scaling_eta_dtn, scaling_eta_r0

def to01_x(T, vmin, vmax):
    return (T - vmin) / (vmax - vmin)

def to01_eta_ext(T, C):
    return C*T

def denormalize(T, vmin, vmax):
    return (vmax - vmin) * T + vmin

### ARCHITECTURES

def ROM1(u, d):
    return psi1(u)

def ROM2(u1d, d, eta, firstiter):
    if firstiter:
        return decoder_0(torch.cat([phi1_0(u1d), phi2_0(d), omega_0(eta)], dim=-1)).mm(V_u0) + clos_0(d)
    else:
        return decoder(torch.cat([phi1(u1d), phi2(d), omega(eta)], dim=-1)).mm(V_u) + (u1d.mean(axis=-1)[0]).unsqueeze(-1) * clos(d)

def ROM3(u, kk):
    x = coarsemap(u)
    #x = torch.where(x.abs() < 1e-3, torch.zeros_like(x), x)
    return x

###################

def interpolate_1d_on_3d(u1d_old, u_interp, idx):
    """
    Interpola u1d_i definita su submeshes1d[i] sulla mesh 3D del sottodominio i.
    """
    # valori corrispondenti
    u_values = u1d_old.vector().get_local()[idx]
    u_interp.vector().set_local(u_values)
    return u_interp





def get_3D_system_RESIDUAL_VERSION(meshV, meshQ, sub_surfaces_tags, LAMBDA, phys_param, nonempty):

    """
    assemble all matrices for the 3d decoupled problem
    A, M, R, L
    
    with boundary conditions

    nonempty == 1 if 3d voxel contain a graph; 0 otherwise 

    """
       
    # phy constant
    # mm - bar
    R  = phys_param['r']

    K1 = Constant((phys_param['rho_int']*phys_param['Kt'])/(phys_param['mu_int']))
    J  = Constant(2*np.pi*phys_param['L_cap']*phys_param['rho_int']*R) #R is used for scaling     
    P =  Constant(phys_param['deltaP'])
    
    BETA = Constant(phys_param['beta']) # VALUE TO BE ADJUSTED <--physical constant beta (see DD-ROM latex)
    p0   = Constant(0) #  VALUE TO BE ADJUST per ora scelto a caso 

    if nonempty == 1:
          # FE spaces
          Q = FunctionSpace(meshQ, 'CG', 1)
          V = FunctionSpace(meshV, 'CG', 1)
             
          dofmap = V.dofmap()
          V_DOF = dofmap.global_dimension()
             
          W = [V, Q]
             
          u, p = map(TrialFunction, W)
          v, q = map(TestFunction, W)
             
          # Computing average operator #
          # Average (R > 0) or trace (R = 0)
          if R > 0:
              # Averaging surface
              cylinder = Circle(radius=R, degree=10)
              Ru, Rv   = Average(u, meshQ, cylinder), Average(v, meshQ, cylinder)
              C        = average_3d1d_matrix(V, Q, cylinder)
          else:
              Ru, Rv = Average(u, meshQ, None), Average(v, meshQ, None)
              C      = trace_3d1d_matrix(V, Q, meshQ)
             
             
          # Line integral                                                                                                                                                 
          dx_ = Measure('dx', domain=meshQ)
             
          # Physical boundary surface integral
          #ds_phy = Measure('ds', domain = meshV, subdomain_data = 111)

          # Virtual (DD) boundary surface integral
          #ds_gammaij = Measure('ds', domain = meshV, sub_surfaces_tags = 222)

          # boundary surface integral
          ds_loc = Measure('ds', domain = meshV, subdomain_data = sub_surfaces_tags)


          # Physical boundary surface integral tag
          physical = 111

          # Virtual (DD) boundary surface integral tag
          virtual = 222



          # We're building decoupled 3D problem with Robin b.c. #
          
          # Laplacian    
          a = block_form(W, 2)
          a[0][0] = K1 * inner(grad(u), grad(v)) * dx 
 
          # Reaction (from Robin bc) on physical boundaries
          re_phy = block_form(W, 2)
          re_phy[0][0] = BETA * inner(u, v) * ds_loc(physical) 
 
          # Reaction (from DD coupling consition) on virtual boundaries gammaij's
          re_gammaij = block_form(W, 2)
          re_gammaij[0][0] = LAMBDA * inner(u, v) * ds_loc(virtual) 
 
          # Coupling matrix 
          m = block_form(W, 2)
          m[0][0] =  J* inner(Ru, Rv) * dx_
          m[0][1] = -J* inner(p, Rv) * dx_
             
          # Forcing term "fixed" from DELTA PI on the 1d
          l1d    = block_form(W, 1)
          l1d[0] =  - J * inner(P,Rv) * dx_  
          

          # Forcing term "fixed" from phy boundary condition
          lf    = block_form(W, 1)
          lf[0] = BETA * inner(p0, v)* ds_loc(physical) 
             
            
          A, R_phy, R_gamma, M, L1d, Lf = map(ii_assemble, (a, re_phy, re_gammaij, m, l1d, lf))


    else:
          '''
          simple laplacian problem + robin boundary condition
          '''

          # FE spaces
          Q = FunctionSpace(meshQ, 'CG', 1)
          V = FunctionSpace(meshV, 'CG', 1)
             
          dofmap = V.dofmap()
          V_DOF  = dofmap.global_dimension()
             
          W = [V, Q]
             
          u, p = map(TrialFunction, W)
          v, q = map(TestFunction, W)

            
          # Line integral                                                                                                                                                 
          dx_ = Measure('dx', domain=meshQ)
             
          # Physical boundary surface integral
          #ds_phy = Measure('ds', domain = meshV, subdomain_data = 111)

          # Virtual (DD) boundary surface integral
          #ds_gammaij = Measure('ds', domain = meshV, sub_surfaces_tags = 222)

          # boundary surface integral
          ds_loc = Measure('ds', domain = meshV, subdomain_data = sub_surfaces_tags)


          # Physical boundary surface integral tag
          physical = 111

          # Virtual (DD) boundary surface integral tag
          virtual = 222



          # We're building decoupled 3D problem with Robin b.c. #
          
          # Laplacian    
          a       = block_form(W, 2)
          a[0][0] = K1 * inner(grad(u), grad(v)) * dx 
 
          # Reaction (from Robin bc) on physical boundaries
          re_phy       = block_form(W, 2)
          re_phy[0][0] = BETA * inner(u, v) * ds_loc(physical) 
 
          # Reaction (from DD coupling condition) on virtual boundaries gammaij's
          re_gammaij       = block_form(W, 2)
          re_gammaij[0][0] = LAMBDA * inner(u, v) * ds_loc(virtual) 

          ''' 
          # DUMMY  Coupling matrix 
          m = block_form(W, 2)
          m[0][0] =  Constant(0.0) * dx_
          m[0][1] =  Constant(0.0) * dx_
             
          # DUMMY Forcing term "fixed" from DELTA PI on the 1d
          l1d    = block_form(W, 1)
          l1d[0] =  Constant(0.0)* dx_  
          '''

          # Forcing term "fixed" from phy boundary condition
          lf    = block_form(W, 1)
          lf[0] = BETA * inner(p0, v)* ds_loc(physical) 
             
            
          A, R_phy, R_gamma, Lf = map(ii_assemble, (a, re_phy, re_gammaij, lf))

          # return empty matrices
          M   = None
          L1d = None

    return A, R_phy, R_gamma, M, L1d, Lf, W, V_DOF






def get_R_gamma(meshV, meshQ, sub_surfaces_tags, LAMBDA):

    
    # FE spaces
    Q = FunctionSpace(meshQ, 'CG', 1)
    V = FunctionSpace(meshV, 'CG', 1)
       
    dofmap = V.dofmap()
    V_DOF = dofmap.global_dimension()
       
    W = [V, Q]
       
    u, p = map(TrialFunction, W)
    v, q = map(TestFunction, W)
       
      
       
    # boundary surface integral
    ds_loc = Measure('ds', domain = meshV, subdomain_data = sub_surfaces_tags)


    # Physical boundary surface integral tag
    physical = 111

    # Virtual (DD) boundary surface integral tag
    virtual  = 222

 
    # Reaction (from DD coupling condition) on virtual boundaries gammaij's
    re_gammaij       = block_form(W, 2)
    re_gammaij[0][0] = LAMBDA * inner(u, v) * ds_loc(virtual) 
 
      
    R_gamma = ii_assemble(re_gammaij)
       
    return R_gamma


def solveLocalFom_RESIDUAL_VERSION(SubProblem3d_PETSc, solver, Lv, f1d, W, emptyvoxel):
       
    # assembly 3d decoupled problem for PETSc #
    
    A00_PET       = SubProblem3d_PETSc[0]   
    Rphy_PET      = SubProblem3d_PETSc[1]   
    Rgamma_PET    = SubProblem3d_PETSc[2]   
    M00_PET       = SubProblem3d_PETSc[3]   
    M01_PET       = SubProblem3d_PETSc[4]   
    L1d_PET       = SubProblem3d_PETSc[5]  
    Lf_PET        = SubProblem3d_PETSc[6] 
    shape         = SubProblem3d_PETSc[7] 


    Lv_PET       = PETSc.Vec().createWithArray(Lv, comm=PETSc.COMM_WORLD)
    
    # sum A00 R's and M00
    A_tot_PET = A00_PET + M00_PET + Rphy_PET + Rgamma_PET


    # total forcing term

    # gestisci il caos di emptyvoxel (1 se voxel ripieno di 1D, else 0)
    if emptyvoxel == 1: 

        f1d_PET   = PETSc.Vec().createWithArray(f1d.vector()[:]) #<-- PETSc vector
        f3d_PET   = np.zeros(shape)
        f3d_PET   = PETSc.Vec().createWithArray(f3d_PET)

        # get the extension of 1d on 3d   
        M01_PET.mult(f1d_PET, f3d_PET)  # <--perform the matrix-vector multiplication M_01 * f1d = f3d

    else:
        f3d_PET = Lf_PET.duplicate()
        f3d_PET.set(0.0)

    # get RHS; vettori definiti su tutti interno + boundary
    b_PET = -f3d_PET + L1d_PET + Lf_PET + Lv_PET    
    
    # initialize  unknown
    u     = np.zeros((shape))
    u_PET = PETSc.Vec().createWithArray(u)   
       
    # SOLVE!
    solver.solve(b_PET, u_PET)

      
    # print
    '''
    print ('iterations = ',               solver.getIterationNumber())
    print ('residual = ', '{:.2e}'.format(solver.getResidualNorm()))#  %.2E
    initial_residual = PETSc.Vec.norm(b_PET)
    final_residual = solver.getResidualNorm()
    last_relative_residual = final_residual / initial_residual
    print('relative residual = ', last_relative_residual)
    print ('converge reason = ',          solver.getConvergedReason())
    print ('residuals at each iter = ',   solver.getConvergenceHistory())

    pc = solver.getPC()

    print("KSP type       =", solver.getType())
    print("PC type        =", pc.getType())
    print("PC side        =", solver.getPCSide())
    print("iterations     =", solver.getIterationNumber())
    print("residual norm  =", solver.getResidualNorm())
    print("reason         =", solver.getConvergedReason())

    pc.view()

    #'''

    if solver.getConvergedReason() != 2:
        
         print ('converge reason = ', solver.getConvergedReason())
         #exit() 
       
    u1_npy = np.array(u_PET.getArray())

    '''
    # paraview plot ---------------------------
    u1=Function(W[0])
    u1.vector()[:]=u1_npy
       
    #------------------------------------------
    '''

    return u1_npy  
       

def H1_norm(v):
    return sqrt(assemble((inner(v, v) + inner(grad(v), grad(v))) * dx))



def expand_interface_dofs_cg1(V, interface_dofs):
    """
    Given a list/array of CG1 dof indices, return the union of:
      - the original interface dofs
      - all CG1 dofs in the first cell-neighborhood of those dofs

    Works for serial meshes.
    """
    mesh = V.mesh()
    tdim = mesh.topology().dim()

    # Build vertex <-> cell connectivity
    mesh.init(0, tdim)

    d2v = dof_to_vertex_map(V)
    v2d = vertex_to_dof_map(V)

    topology = mesh.topology()
    v_to_c = topology(0, tdim)

    interface_dofs = set(int(d) for d in interface_dofs)

    # CG1: dof -> vertex
    seed_vertices = set(int(d2v[d]) for d in interface_dofs)

    # Collect all cells touching those vertices
    touched_cells = set()
    for v in seed_vertices:
        for c in v_to_c(v):
            touched_cells.add(int(c))

    # Collect all vertices of those cells
    neigh_vertices = set(seed_vertices)
    for c in touched_cells:
        cell = Cell(mesh, c)
        for v in cell.entities(0):
            neigh_vertices.add(int(v))

    # Back to dofs
    neigh_dofs = set(int(v2d[v]) for v in neigh_vertices)

    return sorted(neigh_dofs)





def to_numpy(vec):
    """
    Return a NumPy array from either a NumPy array or a PETSc Vec.
    """
    if isinstance(vec, np.ndarray):
        return vec

    if isinstance(vec, PETSc.Vec):
        # returns a NumPy view when possible
        return vec.getArray()

    raise TypeError(f"Unsupported type: {type(vec)}")



def tic():
    return time.perf_counter()
     
def toc(name, t0, step_dict=None):
    elapsed = time.perf_counter() - t0
    timings[name] += elapsed
    if step_dict is not None:
        step_dict[name] = step_dict.get(name, 0.0) + elapsed
    return elapsed





#__________________________________________________________________
#__________________________________________________________________
#__________________________________________________________________
#__________________________________________________________________
#__________________________________________________________________
#__________________________________________________________________
#__________________________________________________________________


start_total = time.time()


#-------------------------------------------------------------
#-------------------------------------------------------------
#                DATA FOR SNAPSHOT GENERATION
#-------------------------------------------------------------
#-------------------------------------------------------------

snapshot_data = {
   'u3d_old': {},
   'u3d_coarse': {},
   'u3d_new': {},
   'eta': {},
   'eta_ext': {},
   'rgamma': {},
   'u1d': {},
   'dist': {},
   'r': {},
   'r0': {}
}

selected_kk = list(range(1,5))

#-------------------------------------------------------------
#-------------------------------------------------------------
#-------------------------------------------------------------



# --------------------------------- CLI ---------------------------------------
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-net_name',       type=str,   required=True)
parser.add_argument('-net_folder',     type=str,   required=True)
parser.add_argument('-f1',             type=int,   required=True) # flux rom flag   
parser.add_argument('-f2',             type=int,   required=True) # solver rom flag  
parser.add_argument('-f3',             type=int,   required=True) # coarse rom flag 
parser.add_argument('-mc',             type=int,   required=False, default = 0   )   
parser.add_argument('-alpha',          type=float, required=False, default = -1.0)
parser.add_argument('-test',           type=str,   default = 'dacan')
parser.add_argument('-coarse',         type=int,   required=False, default = 1   )
parser.add_argument('-save_every',     type=int,   default = 10                  ) # for pvd solution printing
args, _ = parser.parse_known_args()

# ROM flags -------
montecarlo = args.mc
flagrom1   = args.f1
flagrom2   = args.f2
flagrom3   = args.f3
ROM_folder = "ROM_DATA_22-06-26"
#------------------


graph_name = args.net_folder + "-" + args.net_name
coarse_net = graph_name 
alpha      = args.alpha
test_name  = args.test
out_base   = f"outputs/{test_name}/"
coarse     = args.coarse


# --------------------- READ JSON PARAM FILES ----------------------------------
solvers_params  = load_parameters_from_file('./inputdata/solvers_params.json')
physical_params = load_parameters_from_file('./inputdata/physical_params.json')
geom_params     = load_parameters_from_file('./inputdata/geom_params.json')

# solver variables
tol1d   = solvers_params['tol_1d']
tol3d   = solvers_params['tol_3d_loc']
tol3d1d = solvers_params['tol_3d1d']

# geom variables
nref    = geom_params['nref']
n_local = geom_params['n_local']
h       = 1./n_local               #<-- each local subdomain is a cube [0,1]^3








if flagrom1:
   #build local ROMs for fluxes
   V_u, V_u0 = loadPOD(ROM_folder) 
   phi1, phi2, omega, decoder = [], [], [], []
   phi1_0, phi2_0, omega_0, decoder_0 = [], [], [], []

if flagrom2:
   #build local ROMs for solutions 
   V_u, V_u0 = loadPOD(ROM_folder) 
   phi1, phi2, omega, decoder = [], [], [], []
   phi1_0, phi2_0, omega_0, decoder_0 = [], [], [], []


# -----------------------------------------------------------------------------
#                           GEOMETRY  --  **LOADING**
# -----------------------------------------------------------------------------
print("\nLoading pre‑computed geometry...")






geom_folder        = Path(f"./outputs/geom_subdivisions/1D_{graph_name}/nref_{nref}/n_local_{n_local}/")
coarse_geom_folder = Path(f"./outputs/geom_subdivisions/1D_{coarse_net}/nref_{nref}/n_local_{n_local}/")

if not geom_folder.exists():
    raise RuntimeError(f"Geometry folder {geom_folder} not found.  Please run make_box_subdivision.py first.")





# ---- global meshes & markers -------------------------------------------------

# 3D mesh + tags
meshV = Mesh()
with XDMFFile(MPI.comm_world, str(geom_folder / 'meshV.xdmf')) as xdmf:
    xdmf.read(meshV)
    V_markers = MeshFunction('size_t', meshV, 2) # facets tags
    xdmf.read(V_markers)

# 1D mesh + tags
meshQ = Mesh()
with XDMFFile(MPI.comm_world, str(geom_folder / 'meshQ.xdmf')) as xdmf:
    xdmf.read(meshQ)
    Q_markers = MeshFunction('size_t', meshQ, 0) # <-- vertex tags
    xdmf.read(Q_markers)


# function spaces
global_V3d = FunctionSpace(meshV, 'CG', 1)
global_Q1d = FunctionSpace(meshQ, 'CG', 1)
shape3d    = len(meshV.coordinates())
shape1d    = len(meshQ.coordinates())

# measure for 1‑D forcing terms (same as before)
ds1d = Measure('ds', domain=meshQ, subdomain_data=Q_markers)


# build distance function  TO BE MOVED OUTSIDE (MAGARI MISURA IL TEMPO DI ESECUZIONE)
dist = getGraphDist(meshQ, meshV)




# unpack geom variables ---------------------------------------

pkl_file = geom_folder / "geom_data.pkl"   

with open(pkl_file, "rb") as f:
    geom_data = pickle.load(f)      # <-- everything is a single dict 

# unpack
global_is_on_boundary     = geom_data["global_is_on_boundary"]
global_boundary_idxsDOF   = geom_data["global_boundary_idxsDOF"]
global_interior_idxsDOF   = geom_data["global_interior_idxsDOF"]
locals_is_on_boundary     = geom_data["locals_is_on_boundary"]
locals_phy_is_on_boundary = geom_data["locals_phy_is_on_boundary"]
El2g_dof_list             = geom_data["El2g_dof_list"]
global_mu_mult            = geom_data["global_mu_mult"]
penalty                   = geom_data["penalty"]
array_dict1d              = geom_data["array_dict1d"]
mappa_indici1d            = geom_data["mappa_indici1d"]
emptyvoxellist            = geom_data["emptyvoxellist"]
neighs_dict               = geom_data["neighs_dict"]
neighs_list               = geom_data["neighs_list"]
global_weight             = geom_data['global_weight']  
mass_global_list          = geom_data['mass_global_list']
mass_global_sum           = geom_data['mass_global_sum']

# make global mask of crosspoint
global_cross_mask = (penalty < 1.)

if coarse:
    # load coarse matrix stuff               
    c2g_idx          =  geom_data ["c2g_idx"      ]    
    P_c2g_CSR        =  geom_data ["P_c2g_CSR"    ]   
    c_vir_mask       =  geom_data ["c_vir_mask"   ]   
    c_int_mask       =  geom_data ["c_int_mask"   ]   
    P_loc_list       =  geom_data ["P_loc_list"   ]
    p_unity_list     =  geom_data ["p_unity_list" ]
    c_l2g_idx_i_list =  geom_data ["c_l2g_idx_i_list"]   
    Cl2g_dof_list    =  geom_data ["C_l2g_list"]      


    # load coarse mesh #
    
    # 3D mesh + tags
    meshV_COARSE = Mesh()
    with XDMFFile(MPI.comm_world, str(coarse_geom_folder / 'meshV_COARSE.xdmf')) as xdmf:
        xdmf.read(meshV_COARSE)
        V_markers_COARSE = MeshFunction('size_t', meshV_COARSE, 2) # facets tags
        xdmf.read(V_markers_COARSE)
    
    
    global_V3d_COARSE        = FunctionSpace(meshV_COARSE, 'CG', 1)
    shape3d_COARSE           = len(meshV_COARSE.coordinates())

    # load reference 1D mesh + tags for COARSE SOLVER 
    meshQ_FIXED = Mesh()
    with XDMFFile(MPI.comm_world, str(coarse_geom_folder / 'meshQ.xdmf')) as xdmf:
        xdmf.read(meshQ_FIXED)
        Q_FIXED_markers = MeshFunction('size_t', meshQ_FIXED, 0) # <-- vertex tags
        xdmf.read(Q_FIXED_markers)

    # measure for 1‑D forcing terms (same as before)
    ds1d_FIXED = Measure('ds', domain=meshQ_FIXED, subdomain_data=Q_FIXED_markers)



    # get interface global idx
    interface_mask_global = global_mu_mult > 1
    interface_dofs_global = np.where(interface_mask_global)[0]
    internal_dofs_global  = np.where(~interface_mask_global)[0]
    
    
   



# ---- sub‑meshes & their face‑tags ------------------------------------------

# load 3d subdomains
submeshes3d       = []
submeshes3d_tags  = []
idx = 0
while (geom_folder / f'submesh3d_{idx}.xdmf').exists():
    m = Mesh()
    with XDMFFile(MPI.comm_world, str(geom_folder / f'submesh3d_{idx}.xdmf')) as xdmf:
        xdmf.read(m)
        tag = MeshFunction('size_t', m, 2)  # face tags were written alongside
        xdmf.read(tag)
    submeshes3d.append(m)
    submeshes3d_tags.append(tag)
    idx += 1



# load 1d subdomains (Q): use None for empty voxels


# dummy Q space used only as size-compatible placeholder (never for geometry)
dummyQ_mesh = UnitIntervalMesh(1)              # 2 vertices -> CG1 has 2 dofs
dummyQ      = FunctionSpace(dummyQ_mesh, 'CG', 1)




submeshes1d = []

n_sub = len(submeshes3d)  # safest: you iterate submeshes3d everywhere later
# (alternatively len(emptyvoxellist) if that's guaranteed aligned)
for idx in range(n_sub):
    if emptyvoxellist[idx] == 1:
        path = geom_folder / f"submesh1d_{idx}.xdmf"
        if not path.exists():
            raise RuntimeError(f"Expected {path} but it does not exist.")

        m = Mesh()
        with XDMFFile(MPI.comm_world, str(path)) as xdmf:
            xdmf.read(m)

        # if file exists but mesh ended up empty, treat as empty
        if m.num_cells() == 0 or m.num_vertices() == 0:
            submeshes1d.append(dummyQ_mesh)
        else:
            submeshes1d.append(m)
    else:
        '''
        ATTENZIONE: se il dominio è vuoti, ci si piazza una dummy mesh 1D; 
        questa non deve essere mai considerata 
        '''

        submeshes1d.append(dummyQ_mesh)




# build local function spaces and nearesr neigborhood map--------------------------------------------
'''
commento: se un ramo è contenuto in un dominio, ma molto vicino ad un altro, 
la soluzione 1D (u1d) non viene riportata nel dominio vicino ---> dato incompleto/non rappresentativo!
'''

locals_V3d = [FunctionSpace(m, 'CG', 1) for m in submeshes3d]

locals_Q1d  = []
locals_idx  = []
u1d_interp  = []

for i in range(len(submeshes3d)):
    Vloc = locals_V3d[i]

    if emptyvoxellist[i] == 1:
        Qloc = FunctionSpace(submeshes1d[i], 'CG', 1)
        # nearest neighbor map 1D dofs -> 3D dofs
        tree = cKDTree(Qloc.tabulate_dof_coordinates())
        _, idx = tree.query(Vloc.tabulate_dof_coordinates())
        locals_idx.append(idx)
    else:
        Qloc = dummyQ
        # no Q: map everything to empt idx
        locals_idx.append(np.array([]))

    locals_Q1d.append(Qloc)
    u1d_interp.append(Function(Vloc))  # always exists (1D-on-3D field), will be 0 for empty Q

#----------------------------------------------------------------------------------------------------







#????????????????????????????????????????????

# TODO funzione distanza da generare fuori, nella sezione della geometria


# build submesh functions for distance
submeshes_dist = []

for i in range(len(submeshes3d)):
    submeshes_dist.append(El2g_dof_list[i].transpose().dot(dist.vector()[:]))

    # TODO convertire qui in ROM data


# build submesh template functions for 3d interpolation of 1d data 
temp_1d_3d = Function(global_V3d)
#for i in range(len(submeshes3d)):
#    temp_1d_3d.append(Function(locals_V3d[i]))


#????????????????????????????????????????????



# -----------------------------------------------------------------------------
#                           GEOMETRY  --  DONE
# -----------------------------------------------------------------------------
print("Geometry loaded.  Proceeding to assembly and solve...\n")





# initialize ROM stuff
if flagrom1 or flagrom2 or flagrom3:
    psi1, phi1, phi2, omega, decoder, phi1_0, phi2_0, omega_0, decoder_0, clos_0, clos = loadROM(locals_V3d[0], ROM_folder)
    coarsemap = MINN_coarse(locals_V3d[0], 10001)


# ---- INITIAL 3‑D/1‑D INITIAL CONDITIONS ---------------------------------------

ubc_1d_in   = physical_params['ubc_1d_in']
ubc_1d_out  = physical_params['ubc_1d_out']
ubc_3d_in, ubc_3d_out = 0, 0

u1d_old = initialize_1d(meshQ, Q_markers, ubc_1d_in,  ubc_1d_out)
u3d_old = initialize_3d(meshV, V_markers, ubc_3d_in, ubc_3d_out)

u1d_loc = assemble_local_fun1d(u1d_old, array_dict1d, mappa_indici1d, submeshes1d, emptyvoxellist)
#---------------------------------------------------------------------------------


# get cKDT projection of 1d data
tottime = 0
for i in range(len(submeshes3d)):
    if emptyvoxellist[i] == 1:
       start = time.time()
       u1d_interp[i] = interpolate_1d_on_3d(u1d_loc[i], u1d_interp[i], locals_idx[i])
       tottime += (time.time() - start) 
print("Time elapsed for projection: " + str(tottime) + " seconds.")







# DEFINE OPTIMIZED PARAMETER LAMBDA

K1     = physical_params['rho_int']*physical_params['Kt']/(physical_params['mu_int'])
print("K1, alpha, h", K1, alpha, h)
LAMBDA = K1 * h**alpha

print(f"-------------------------------------------------------------------\n")
print(f"Optimized Schwarz Method for 3D problem. Robin parameter: {LAMBDA}")
print(f"-------------------------------------------------------------------\n")



from lib.modules_solver3d_loc import *
# importo separatamente perchè ci sono dei conflitti con i submesh dictionary


# assembly global 1d problem
start_1D_assem = time.time()
A1d, M1d, b, W = get_1D_system(meshV, meshQ, ds1d, physical_params)
end_1D_assem   = time.time()
ass1D_time     = end_1D_assem - start_1D_assem #<--TIME

global_1d_problem = [A1d, M1d, b, W]

assembly_timing_table = {}


if coarse:

    # ASSEMBLE COARSE STUFF #
    
    print("get coarse system")
    init_time  = time.time()     
    DECOUPLED_COARSE_SYSTEM, _, _, W_COARSE = get_3D1D_system(meshV_COARSE, 
                                                                     meshQ_FIXED, 
                                                                     ds1d_FIXED, 
                                                                     physical_params)

    print(f"global coarse assemble time:", time.time()-init_time) #<-- measuring single time     ; to be removed

    print("get coarse system...done")
    #-------------------------------------------------------------------------------------#





# assembly subdomains 3d-decoupled problem
'''
DO MPI HERE!...TODO
'''


SubProblems3d          = [] # <-- collection of local problems
SubProblems3d_PETSc    = [] # <-- collection of local problems
SubProblems3d_PETSc_F1 = [] # <-- collection of local problems
Ws                     = [] # <---collection of function spaces [V_sub, Q_sub]
subdomain_solvers      = [] # <---collection of solvers


if flagrom1 == 1:
    #get only the Rgamma matrix
    for i in range(len(submeshes3d)):


        curr_mesh3d     = submeshes3d[i]
        curr_mesh1d     = submeshes1d[i]
        curr_bound_tags = submeshes3d_tags[i]
         
        R_gamma = get_R_gamma(curr_mesh3d, curr_mesh1d, curr_bound_tags, LAMBDA)

        '''
        probably here, the standar poisson matrix should be assembled
        '''

        shape   = len(curr_mesh3d.coordinates())
        # matrix to PETSc
        Rgamma_PET = as_backend_type(R_gamma[0][0]).mat()     #<-- PETSc matrix
    
        SubProblems3d_PETSc_F1.append((Rgamma_PET,shape))
 



if flagrom1 == 0 or flagrom2 == 0: 

    #-------------------------------------
    # LOOP ON SUBDOMAIN FOR ASSEMBLE STUFF
    #-------------------------------------

    start_local_assem = time.time()
    
    for i in range(len(submeshes3d)):

        curr_mesh3d     = submeshes3d[i]
        curr_mesh1d     = submeshes1d[i]
        curr_bound_tags = submeshes3d_tags[i]
    
        A, R_phy, R_gamma, M, L1d, Lf, W_loc, _ = get_3D_system_RESIDUAL_VERSION(curr_mesh3d, curr_mesh1d, curr_bound_tags, LAMBDA, physical_params, emptyvoxellist[i])
    
        SubProblems3d.append((A, R_phy, R_gamma, M, L1d, Lf))
        Ws.append(W_loc)
    
        # the same subproblem collection, but in PETSc format #
    
        # matrices to PETSc

        A00_PET         = as_backend_type(A[0][0]).mat()           #<-- PETSc matrix
        Rphy_PET        = as_backend_type(R_phy[0][0]).mat()       #<-- PETSc matrix
        Rgamma_PET      = as_backend_type(R_gamma[0][0]).mat()     #<-- PETSc matrix
       


        Lf_PET          = as_backend_type(Lf[0])                   #<-- forcing therm from physica Robin
        shape           = Lf_PET.size()                            #<--get the vector 3d vector shape
        Lf_PET          = Lf_PET.vec()
 
        if emptyvoxellist[i] == 1:

             '''
             flag empty coupling matrix and 1D forcing term
             if the voxed has empty 1D subdomain
             '''

             M00_PET         = ii_convert(M[0][0]).mat()                #<-- PETSc matrix
             M01_PET         = ii_convert(M[0][1]).mat()                #<-- PETSc matrix
             L1d_PET         = ii_convert(L1d[0]).vec()                 #<-- forcing therm from DELTApi
        else:
             M01_PET         = None  
             L1d_PET = Lf_PET.duplicate()
             L1d_PET.set(0.0)
             M00_PET = A00_PET.duplicate(copy=True)                     # same sparsity + size
             M00_PET.zeroEntries()                                      # set all values to 0
   
   

        SubProblems3d_PETSc.append((A00_PET,
                                    Rphy_PET,
                                    Rgamma_PET,
                                    M00_PET,
                                    M01_PET,
                                    L1d_PET,
                                    Lf_PET,
                                    shape)
                                   )


        if flagrom2 == 0:
            '''
            # Build the collection of SOLVERS #
            A_tot_PET = A00_PET + M00_PET + Rphy_PET + Rgamma_PET
    
            # Create a PETSc Krylov solver
            solver = PETSc.KSP().create()
            solver.setOperators(A_tot_PET)
            solver.setType(PETSc.KSP.Type.GMRES) # TODO: to be switched to CG ---> enforce symmetric AMG
            solver.setTolerances(rtol=tol3d)
            solver.setPCSide(1)
            solver.view()
            solver.setFromOptions()  # <--Allow setting options from the command line or a file
    
            # set PC #
            pc = solver.getPC()
            pc.setType(PETSc.PC.Type.HYPRE)
            pc.setHYPREType("boomeramg")
            # Access the options database
            opts = PETSc.Options()
    
            # Set the number of iterations
            opts['pc_hypre_boomeramg_max_iter'] = 1
            # Set the maximum number of AMG levels
            opts['pc_hypre_boomeramg_max_levels'] = 3
            # Set ILU type to none
            opts['pc_factor_mat_solver_package'] = 'none'
    
            subdomain_solvers.append(solver)
            '''


            A_tot_PET = A00_PET + M00_PET + Rphy_PET + Rgamma_PET
            A_tot_PET.assemble()
            
            solver = PETSc.KSP().create(comm=PETSc.COMM_WORLD)
            solver.setOperators(A_tot_PET)
            
            solver.setType(PETSc.KSP.Type.CG)
            solver.setTolerances(rtol=tol3d)
            solver.setPCSide(PETSc.PC.Side.LEFT)
            
            pc = solver.getPC()
            pc.setType(PETSc.PC.Type.HYPRE)
            pc.setHYPREType("boomeramg")
            
            opts = PETSc.Options()
            opts["pc_hypre_boomeramg_max_iter"] = 1
            #opts["pc_hypre_boomeramg_max_levels"] = 20
            # Set ILU type to none
            opts['pc_factor_mat_solver_package'] = 'none'
            
            solver.setFromOptions()
            
            subdomain_solvers.append(solver)
            #-----------------------------------------------------------------------
    
    
    
    end_local_assem = time.time()
    ass_loc_time = end_local_assem - start_local_assem #<--TIME









# initialize list
u3d_fun_new           = [] # <--collection of local 3d function dof ordered
u3d_val_new           = [] # <--collection of local 3d function vertex ordered

stagnation_history_3D = [] # <-- relative stagnation criterion
refL2_history_3D      = [] # <-- relative L2 convergence to  REFERENCE solution
refH1_history_3D      = [] # <-- relative H1 convergence to  REFERENCE solution
refL_inf_history_3D   = [] # <-- relative L_inf convergence to  REFERENCE solution


L2_REFNORM_history_3D      = [] # <--  L2 of  REFERENCE solution
H1_REFNORM_history_3D      = [] # <--  H1 of  REFERENCE solution
L_inf_REFNORM_history_3D   = [] # <--  L_inf of  REFERENCE solution
L2_REFNORM_history_1D      = [] # <--  L2 of  REFERENCE solution
H1_REFNORM_history_1D      = [] # <--  H1 of  REFERENCE solution
L_inf_REFNORM_history_1D   = [] # <--  L_inf of  REFERENCE solution



stagnation_history_1D = [] # <-- relative stagnation criterion
refL2_history_1D      = [] # <-- relative L2 convergence to  REFERENCE solution
refH1_history_1D      = [] # <-- relative H1 convergence to  REFERENCE solution
refL_inf_history_1D   = [] # <-- relative L_inf convergence to  REFERENCE solution

L2jump_history               = [] # <-- L2 jump norm on Gamma
scaled_L2jump_history        = [] # <-- h-scaled  L2 jump norm on Gamma
DG_norm_history              = [] # <-- DG-type norm

window_stagnation     = []

corr_internal_history   = []     
corr_interface_history  = [] 
corr_tot_history        = [] 

if  montecarlo == 0:



    '''
    # --LOAD MONOLYTHIC REFERENCE SOLUTION ------------
    ref_solution_path = f"outputs/REFERENCE_SOLUTION_outputs/{graph_name}/{nref}-{n_local}/"
    
    # load dof-ordered vector
    u_REF_3d_np = np.load(f"{ref_solution_path}{graph_name}_net_sol3d_tol{tol3d1d}.npy")
    u_REF_1d_np = np.load(f"{ref_solution_path}{graph_name}_net_sol1d_tol{tol3d1d}.npy")
    
    #function space
    u_REF_3d = Function(W[0])
    u_REF_3d.vector()[:] = u_REF_3d_np
    u_REF_1d = Function(W[1])
    u_REF_1d.vector()[:] = u_REF_1d_np
    
    # print in the test folder
    reference_folder = f"outputs/{test_name}/1D_{graph_name}/nref_{nref}/n_local_{n_local}/"
    
    u_REF_3d.rename("u","u")
    u_REF_1d.rename("u1d","u1d")
    
    File(f"{reference_folder}REF_SOL_3D.pvd") << u_REF_3d
    File(f"{reference_folder}REF_SOL_1D.pvd") << u_REF_1d
    #--------------------------------------
    '''

    # --LOAD DDFOM REFERENCE SOLUTION ------------
    
    ref_solution_path = f"outputs/REFERENCE_DD-FOM/1D_{graph_name}/nref_{nref}/n_local_{n_local}/alpha_-1.0/"

    # load dof-ordered vector
    u_REF_3d_np = np.load(f"{ref_solution_path}REF_3d_sol.npy")
    u_REF_1d_np = np.load(f"{ref_solution_path}REF_1d_sol.npy")
    
    #function space
    u_REF_3d = Function(W[0])
    u_REF_3d.vector()[:] = u_REF_3d_np
    u_REF_1d = Function(W[1])
    u_REF_1d.vector()[:] = u_REF_1d_np
    
    # print in the test folder
    reference_folder = f"outputs/{test_name}/1D_{graph_name}/nref_{nref}/n_local_{n_local}/"
    
    u_REF_3d.rename("u","u")
    u_REF_1d.rename("u1d","u1d")
    
    File(f"{reference_folder}REF_SOL_3D.pvd") << u_REF_3d
    File(f"{reference_folder}REF_SOL_1D.pvd") << u_REF_1d
    #--------------------------------------





nmax              = 20    # Maximum number of DD-FOM iterations
kk                = 0
stagnation_window = 10
conv_tol          = 1e-4

# outputs folder
out_folder = f"outputs/{test_name}/1D_{graph_name}/nref_{nref}/n_local_{n_local}/alpha_{alpha}/"



# bounds for ROM
u3d_old_min, u3d_old_max, u3d_new_min, u3d_new_max, d_min, d_max, u1d_min, u1d_max, scaling_eta_ext, scaling_eta_dtn, scaling_eta_r0 = loadbounds(ROM_folder)


mask_interface = (global_mu_mult == 2) # valid for 3D domain
mask_cross     = (global_mu_mult > 2)  # valid for 3D domain


#-----------------------------------------
#             START LOOP
#-----------------------------------------
BREAKLOOP   = 0
start_SOLVE = time.time()


if coarse:
    print("\n")
    print(" \nCOARSE CORRECTION GLOBAL\n ")
    print("\n")


while( kk < nmax):

    print("step kk = ", kk , "\n") 


    if kk == 0:
        firstiter = 1
    else:
        firstiter = 0
        #flagrom2  = 1


    if coarse:


        #-------------------------------------------#
        #                                           # 
        #            COARSE CORRECTION              #
        #                                           #
        #-------------------------------------------#

        # starting from u3d_old u1d_old #


        global_residual = np.zeros(global_V3d.dim())     
        u3d_old_loc, _  = local_solution3d(submeshes3d, u3d_old) # <--- create local solutions from old step

        '''
        Get the residual
        '''

        if flagrom3 == 0:
             '''
             get the fine residual exact
             '''
             if kk == 0:
                 print("COARSE CORRECTION WITH EXACT RESIDUAL")

             '''
             # ------------------------------------------------------------
             # unpack operators      
             # ------------------------------------------------------------
             A = DECOUPLED_FULL_SYSTEM[0]     
             M = DECOUPLED_FULL_SYSTEM[1] 
             # ------------------------------------------------------------
             # convert FEniCS blocks to PETSc matrices
             # ------------------------------------------------------------
             A00_PET = as_backend_type(A[0][0]).mat()
             M00_PET = ii_convert(M[0][0]).mat()
             M01_PET = ii_convert(M[0][1]).mat()
             # total 3D operator: A00 + M00
             A_tot_PET = A00_PET.copy()
             A_tot_PET.axpy(1.0, M00_PET)
             A_tot_PET.assemble()            # for safety
             # ------------------------------------------------------------
             # vectors               
             # ------------------------------------------------------------
             f1d_PET      = PETSc.Vec().createWithArray(u1d_old.vector()[:])
             u_3d_old_PET = PETSc.Vec().createWithArray(u3d_old.vector()[:])
                                     
             # 3D forcing term       
             Lf_backend = ii_convert(b_FULL[0])
             Lf_PET     = Lf_backend.vec()
                                     
             # ------------------------------------------------------------
             # build b = Lf - M01*f1d
             # ------------------------------------------------------------
             f3d_PET = A_tot_PET.createVecRight()
             M01_PET.mult(f1d_PET, f3d_PET)
                                     
             b_PET = Lf_PET.copy()   
             b_PET.axpy(-1.0, f3d_PET)
                                     
             # ------------------------------------------------------------
             # residual r = b - A_tot*u_old
             # ------------------------------------------------------------
             Au_PET = A_tot_PET.createVecRight()
             A_tot_PET.mult(u_3d_old_PET, Au_PET)
                                     
             residual = b_PET.copy()    
             residual.axpy(-1.0, Au_PET)
             res_np = to_numpy(residual) 
             '''
             # get the problem global residual-------------------------------


             for i in range (len(submeshes3d)):
                   '''
                   Get the residual  F_i - (A + R + M)_i * u_i
                   get the residual in the exact (decomposed) way 
                   '''

                   SubProblem3d_PETSc = SubProblems3d_PETSc[i] #<-- pick i-esim subproblem

                   A00_PET       = SubProblem3d_PETSc[0]
                   Rphy_PET      = SubProblem3d_PETSc[1]
                   M00_PET       = SubProblem3d_PETSc[3]
                   M01_PET       = SubProblem3d_PETSc[4]
                   L1d_PET       = SubProblem3d_PETSc[5]
                   Lf_PET        = SubProblem3d_PETSc[6]
                   shape         = SubProblem3d_PETSc[7]


                   # vectors
                   u3d_PET = PETSc.Vec().createWithArray(u3d_old_loc[i].vector()[:]) #<-- PETSc vector of 1d solution

                   if emptyvoxellist[i] == 1:

                        f1d_PET      = PETSc.Vec().createWithArray(u1d_loc[i].vector()[:]) #<-- PETSc vector of 1d solution

                        # graph forcing term
                        f3d_PET   = np.zeros(shape)
                        f3d_PET   = PETSc.Vec().createWithArray(f3d_PET)


                        # get the extension of 1d on 3d
                        M01_PET.mult(f1d_PET, f3d_PET)  # <--perform the matrix-vector multiplication M_01 * f1d = f3d


                   else:
                       f3d_PET = Lf_PET.duplicate()
                       f3d_PET.set(0.0)


                   AAA     = (A00_PET + M00_PET + Rphy_PET)
                   tmp_AAA = AAA.createVecLeft()
                   AAA.mult(u3d_PET, tmp_AAA)

                   local_residual = -f3d_PET + L1d_PET + Lf_PET \
                               - tmp_AAA \

                   global_residual += El2g_dof_list[i].dot(local_residual)


             #-------------

             
             #solve and collect training data r0_list


             e, r_list, r0_list = getCOARSEcorrection(DECOUPLED_COARSE_SYSTEM, 
                                                     P_loc_list,
                                                     p_unity_list,
                                                     El2g_dof_list,
                                                     Cl2g_dof_list,              
                                                     global_residual, 
                                                     global_V3d, 
                                                     tol3d1d
                                                  )

        elif flagrom3 == 1:
             '''
             get the coarse residual as sum of surrogate coarse local residual
             '''
             if kk == 0:
                 print("COARSE CORRECTION WITH SURROGATE COARSE RESIDUAL")

             coarse_res_np = np.zeros((shape3d_COARSE ))  # this has the size of the GLOBAL coarse space. this is accessory and can be avoided by building the right map from local and global coarse space...if you need it, use the list "c_l2g_idx_i_list" mapping local i coarse to global coarse dof

             if kk == 0:
                 coarsemap.load(ROM_folder + "/coarsemap_0.npz")
             elif kk == 1:
                 coarsemap.load(ROM_folder + "/coarsemap_1.npz")
             elif kk == 2:
                 coarsemap.load(ROM_folder + "/coarsemap_k.npz")

             # better here to parallelize with batch
             
             for i in range (len(submeshes3d)):


                       u3d_old_torch                = GPU.tensor(to01_x(u3d_old_loc[i].vector()[:], u3d_old_min, u3d_old_max)).unsqueeze(0)

                       if kk == 0:
                            coarse_local_residual_torch  = (ROM3(u3d_old_torch, kk) / (-scaling_eta_r0)).flatten()
                       else:
                            coarse_local_residual_torch  = (ROM3(u3d_old_torch, kk) / scaling_eta_r0).flatten()

                       # TODOOOO: in questa somma va inserita la matrice Cl2g_dof     
                       start = time.time()
                       coarse_res_np[Cl2g_dof_list[i].nonzero()[0]] += coarse_local_residual_torch.cpu().detach().numpy()
                       end = time.time() - start

             # solve
     
             e = getSURROGATECOARSEcorrection(DECOUPLED_COARSE_SYSTEM, 
                                               P_loc_list,
                                               p_unity_list,
                                               El2g_dof_list,
                                               coarse_res_np, 
                                               global_V3d, 
                                               tol3d1d
                                               )





        # correct_u_3d_old #

        u3d_old_corrected             = Function(W[0])
        u3d_old_corrected.vector()[:] = u3d_old.vector()[:]


      
        # set_correction GLOBALLY

        tmp    = u3d_old_corrected.vector()[:]
        tmp   += e.vector()

        u3d_old_corrected.vector()[:] = tmp


        # collect norm
        e_vec = e.vector()[:]
        
        corr_interface  = np.linalg.norm(e_vec[interface_dofs_global])
        corr_internal   = np.linalg.norm(e_vec[internal_dofs_global])
        corr_tot        = np.linalg.norm(e_vec)


        corr_internal_history.append(corr_internal)
        corr_interface_history.append(corr_interface)
        corr_tot_history.append(corr_tot)




        # GET NEW LOCAL 3D SOLUTION + CORRECTION 
        u_loc3d, u_loc3d_val = local_solution3d(submeshes3d, u3d_old_corrected)            # <--- create local solutions (old step)

        u3d_old.vector()[:] = u3d_old_corrected.vector()[:]
 

        #-------------------------------------------#
        #                                           # 
        #         END COARSE CORRECTION             #
        #                                           #
        #-------------------------------------------#

    else:

        # GET NEW LOCAL 3D SOLUTION 
        # 3D --> questo passaggio si può ottimizzare con le matrici di estensione
        u_loc3d, u_loc3d_val = local_solution3d(submeshes3d, u3d_old)            # <--- create local solutions (old step)



    # get the inverse (i.e. with a minus sign ) flux of every domain at old step ------------------

    locals_eta      = []  #<-- define list of fluxes
    locals_dtn_eta  = []  #<-- define list of DtN fluxes
    dirich_vectors  = []  #<-- define list of dirichlet data


    #??????????????????
    totalrom1     = 0
    totalfluxtime = 0
    #??????????????????




    '''
    DO MPI HERE!...TODO
    '''


    
    if flagrom1 == 0:
        if kk == 0: 
           print("EXACT FLUXES")
        for i in range(len(submeshes3d)):
            '''
            inverse flux is  F_j - (A + R + M)_J * u_j
                                   + LAMBDA*R_j * u_j
            '''
            local_shape = submeshes3d[i].coordinates().shape

            SubProblem3d_PETSc = SubProblems3d_PETSc[i] #<-- pick i-esim subproblem

            A00_PET       = SubProblem3d_PETSc[0]
            Rphy_PET      = SubProblem3d_PETSc[1]
            Rgamma_PET    = SubProblem3d_PETSc[2]
            M00_PET       = SubProblem3d_PETSc[3]
            M01_PET       = SubProblem3d_PETSc[4]
            L1d_PET       = SubProblem3d_PETSc[5]
            Lf_PET        = SubProblem3d_PETSc[6]
            shape         = SubProblem3d_PETSc[7]


            # vectors
            u3d_PET = PETSc.Vec().createWithArray(u_loc3d[i].vector()[:]) #<-- PETSc vector of 1d solution

            if emptyvoxellist[i] == 1:

                 f1d_PET      = PETSc.Vec().createWithArray(u1d_loc[i].vector()[:]) #<-- PETSc vector of 1d solution

                 # graph forcing term
                 f3d_PET   = np.zeros(shape)
                 f3d_PET   = PETSc.Vec().createWithArray(f3d_PET)


                 # get the extension of 1d on 3d
                 M01_PET.mult(f1d_PET, f3d_PET)  # <--perform the matrix-vector multiplication M_01 * f1d = f3d


            else:
                f3d_PET = Lf_PET.duplicate()
                f3d_PET.set(0.0)



            # GET ROBIN FLUX #

            '''# get the flux that live on the virtual boundary
            vir_eta  = - f3d_L + L1d_L + Lf_L                \
                       - D_LI.dot(u3d_I) - D_LL.dot(u3d_L)   \
                       + Rgamma_PET_LL.dot(u3d_L)'''

            D = (A00_PET + M00_PET + Rphy_PET)



            # get the flux that lives on the virtual boundary
            tmp_D = D.createVecLeft()
            D.mult(u3d_PET, tmp_D)


            tmp_Rgamma = Rgamma_PET.createVecLeft()
            Rgamma_PET.mult(u3d_PET, tmp_Rgamma)

            local_eta = -f3d_PET + L1d_PET + Lf_PET \
                        - tmp_D \
                        + tmp_Rgamma



            locals_eta.append(local_eta)

            if montecarlo:
                  local_dtn_eta = -f3d_PET + L1d_PET + Lf_PET - tmp_D  #assemble only neuman part of the robin flux
                  dirich_vectors.append(tmp_Rgamma)
                  locals_dtn_eta.append(local_dtn_eta)

        #--------------------------------------------------------------------





    if flagrom1 == 1:
        if kk == 0: 
            print("SURROGATE FLUXES")
        for i in range (len(submeshes3d)):
            '''
            inverse flux is  F_j - (A + R + M)_J * u_j
                                   + LAMBDA*R_j * u_j
            '''
            local_shape = submeshes3d[i].coordinates().shape

            SubProblem3d_PETSc_F1 = SubProblems3d_PETSc_F1[i] #<-- pick i-esim subproblem

            Rgamma_PET    = SubProblem3d_PETSc_F1[0]
            shape         = SubProblem3d_PETSc_F1[1]


            # vectors
            u3d_PET      = PETSc.Vec().createWithArray(u_loc3d[i].vector()[:]) #<-- PETSc vector of 1d solution

            # get the LAMBDA part 

            tmp_Rgamma = Rgamma_PET.createVecLeft()
            Rgamma_PET.mult(u3d_PET, tmp_Rgamma)

            # fill the empty pthon array with the PETSc vector 
            Rgamma_loc           = np.zeros((local_shape[0])) #<-- array defined on subdomain
            Rgamma_loc[:]        = tmp_Rgamma
            
            u3d_old_torch        = GPU.tensor(to01_x(u_loc3d[i].vector()[:], u3d_old_min, u3d_old_max)).unsqueeze(0)
            dist_loc             = GPU.tensor(submeshes_dist[i]).unsqueeze(0)

            startrom1            = time.time()
            locals_eta_torch     = (ROM1(u3d_old_torch, dist_loc) / scaling_eta_dtn).flatten() + GPU.tensor(Rgamma_loc)
            endrom1              = time.time() - startrom1


            totalrom1            = totalrom1 + endrom1
            locals_eta.append(locals_eta_torch.cpu().detach().numpy())
            #set_trace()
            if montecarlo:
                local_dtn_eta_torch = (ROM1(u3d_old_torch, dist_loc) / scaling_eta_dtn).flatten().cpu().detach().numpy() # only neumann flux data
                locals_dtn_eta.append(local_dtn_eta_torch)
                dirich_vectors.append(Rgamma_loc)                                                 # only dirich data

        #--------------------------------------------------------------------
    '''
    if flagrom1:
        print("End of surrogate modeling of Dirichlet to Neumann map. Time elapsed: " + str(totalrom1) + " seconds.\n")
    '''

    u3d_fun_new.clear()
    u3d_val_new.clear()



    # LOOP ON SUBDOMAINS #

    totalrom2      = 0
    totaltime      = 0
    totalimport    = 0
    local_flux_fun = []

    for i in range (len(submeshes3d)):

        u_loc3d[i].rename("u", "u")
        #File(f'outputs/u_loc_{i}.pvd') << u_loc3d[i], kk

        curr_mesh3d     = submeshes3d[i]
        curr_mesh1d     = submeshes1d[i]
        V_loc           = locals_V3d[i]
        shape           = len(V_loc.tabulate_dof_coordinates())
        zeros           = np.zeros(shape)
        d               = submeshes_dist[i] 



        # get robin-flux forcing term (forcing term deriving from DD robin coupling)
        i_local_eta   = np.zeros(V_loc.tabulate_dof_coordinates().shape[0])
        eta_global_in = np.zeros(global_V3d.dim())

        #for j in neighs_list[i]:
        for j, _ in enumerate(submeshes3d):
            if j != i:
                #i_local_eta[neighs_dict[(i,j)][0]] = locals_eta[j][neighs_dict[(i,j)][1]]
                
                # get local mask
   
                local_mask_cross = El2g_dof_list[j].astype(bool).transpose().dot(mask_cross)

                if isinstance(locals_eta[j], PETSc.Vec):
                     eta_j            = locals_eta[j].getArray(readonly=True) 

                elif isinstance(locals_eta[j], np.ndarray):
                     eta_j            = locals_eta[j] 

                # interface incoming flux
                eta_j_global                   = El2g_dof_list[j].dot(eta_j)
                eta_global_in[mask_interface] += eta_j_global[mask_interface]

                #do this: sum_j != i penalty  * (A_i * (\eta_j/A_j)) ....IT WORKS! NO MORE CROSS POINT ARTIFACTS             
                normalized_local_j_flux                    = np.zeros_like(eta_j)
                normalized_local_j_flux[local_mask_cross]  = eta_j[local_mask_cross]/El2g_dof_list[j].transpose().dot(mass_global_list[j])[local_mask_cross]
                normalized_global_j_flux                   = El2g_dof_list[j].dot(normalized_local_j_flux)



                eta_global_in[mask_cross]  += mass_global_list[i][mask_cross] * normalized_global_j_flux[mask_cross] * penalty[mask_cross]

                              
        i_local_eta[:] = El2g_dof_list[i].transpose().dot(eta_global_in) 


        #--------------------- 



        #---------------------------


        if flagrom2 == 0: 
            if kk == 0: 
                print("EXACT SOLVER")
            # PETSc 3D LOCAL SOLVER #

            u3d_loc = solveLocalFom_RESIDUAL_VERSION(
                                                     SubProblems3d_PETSc[i],
                                                     subdomain_solvers[i],
                                                     i_local_eta,
                                                     u1d_loc[i],
                                                     Ws[i],
                                                     emptyvoxellist[i]
                                                     )


            if montecarlo:
               # collect the aggregated robin fluxes on subdomain
               local_flux_fun = i_local_eta


            u3d_fun_loc = Function(V_loc)
            u3d_fun_loc.vector()[:] = u3d_loc
            u3d_fun_loc.rename("u", "u")
            u3d_fun_new.append(u3d_fun_loc)
            u3d_val_new.append(u3d_fun_loc.compute_vertex_values())
            '''
            # plot----------------------------------------------------
            File(f'{out_folder}u_lok_{i}_{kk}.pvd')  << u3d_fun_loc, kk
            #----------------------------------------------------------
            ''' 





        if flagrom2 == 1:
            if kk == 0: 
                print("SURROGATE SOLVER")
            # ROM 3D LOCAL SOLVER # 
            startromming2 = time.time()
            u1d        = GPU.tensor((u1d_interp[i].vector()[:] - 0.05) / 0.02).unsqueeze(0)
            dist_loc   = GPU.tensor(d).unsqueeze(0)
            eta_ext    = GPU.tensor(to01_eta_ext(i_local_eta, scaling_eta_ext)).unsqueeze(0)

            endromming2 = time.time() - startromming2
            totalimport = totalimport + endromming2

            startrom2 = time.time()

            u3d_loc_torch = ROM2(u1d, dist_loc, eta_ext, firstiter).flatten()
            endrom2 = time.time() - startrom2

            totalrom2 = totalrom2 + endrom2

            startromming3 = time.time()
            u3d_loc = (u3d_new_max - u3d_new_min)*(u3d_loc_torch.cpu().detach().numpy()) + u3d_new_min

            endromming = time.time() - startromming3

            totaltime = totaltime + endromming

            u3d_fun_loc = Function(V_loc)
            u3d_fun_loc.vector()[:] = u3d_loc
            u3d_fun_loc.rename("u", "u")
            u3d_fun_new.append(u3d_fun_loc)
            u3d_val_new.append(u3d_fun_loc.compute_vertex_values())
            
            #if kk == 1 and i==5:
            #    import pdb; pdb.set_trace()
            # plot----------------------------------------------------
            #File(f'./outputs/plotDDROM/u3d_new_{i}_{kk}.pvd')  << u3d_fun_loc, kk
            #----------------------------------------------------------


        #------------------------------------------------------------#
        #------------------------------------------------------------#
        #                                                            # 
        # -----------------   MONTECARLO FLAG -----------------------#
        #                                                            #
        #------------------------------------------------------------#
        #------------------------------------------------------------#


        # Assemble Robin transmission contribution for updated solution in Montecarlo simulations

        if montecarlo and ((kk+1 in selected_kk) or BREAKLOOP == 1):

            V  = FunctionSpace(curr_mesh3d, 'CG', 1)  # placeholder, sovrascrivi con la mesh corretta
            Q  = FunctionSpace(curr_mesh1d, 'CG', 1)  # placeholder, sovrascrivi con la mesh corretta
            f  = Function(V)
            f1 = Function(Q)


            for key, val in {
                'u3d_old'   : u_loc3d[i].vector()[:],     # soluzione 3D al passo kk
                'u3d_coarse': u3d_old_loc[i].vector()[:],  # soluzione 3D non corretta al passo kk
                'u3d_new'   : u3d_loc,                    # soluzione 3D al passo kk+1
                'eta'       : locals_dtn_eta[i],          # flusso NEUMANN associato al sottodominio
                'eta_ext'   : local_flux_fun,             # flusso di ROBIN AGGREGATO sul sottominio
                'rgamma'    : dirich_vectors[i],          # dato di DIRICHLET
                'u1d'       : u1d_interp[i].vector()[:],  # interpolazione della soluzione 1D sui sottodomini 3D
                'dist'      : submeshes_dist[i],          # funzione distanza   
                'r'         : r_list[i],                  # residuo locale coarse
                'r0'        : r0_list[i]                  # residuo locale coarse iniziale
            }.items():
                if i not in snapshot_data[key]:
                    snapshot_data[key][i] = []
                snapshot_data[key][i].append(val)
 
                if key == 'u3d_new':
                    f.vector()[:] = val
                    f.rename(key, key)
                    File(os.path.join(out_base, f"snap_folder/{key}_sub{i}_g{graph_name}_snap{kk}.pvd")) << f




        #------------------------------------------------------------#
        #------------------------------------------------------------#
        #------------------------------------------------------------#
        #------------------------------------------------------------#
        #------------------------------------------------------------#
        #------------------------------------------------------------#







    # assembly global 3d solution
    global_3d_sol = np.zeros(shape3d)
    
    for i in range (len(submeshes3d)):
        global_3d_sol += El2g_dof_list[i].dot(u3d_fun_new[i].vector()[:])

    global_3d_sol = global_3d_sol/global_mu_mult

    u3d_global_fun             = Function(global_V3d)
    u3d_global_fun.vector()[:] = global_3d_sol




    # SOLVE GLOBAL 1D
    if  kk%5 == 0:
        u1d_new = solve_1d(global_1d_problem, u3d_global_fun, tol1d)
    else:
        u1d_new = u1d_old.copy()

    if not montecarlo:
        # CONTROL CRITERIA #

        # check iteration stagnation
        stagnation_3D            = np.linalg.norm( u3d_global_fun.vector()[:] - u3d_old.vector()[:] ) / np.linalg.norm( u3d_old.vector()[:] )
        stagnation_history_3D.append(stagnation_3D) # <-- relative stagnation criterion

        stagnation_1D            = np.linalg.norm( u1d_new.vector()[:] - u1d_old.vector()[:] ) / np.linalg.norm( u1d_old.vector()[:] )
        stagnation_history_1D.append(stagnation_1D) # <-- relative stagnation criterion



        # check error with respect 3D solution #


        # relative L2 error

        L2e_3D = sqrt(assemble(inner(u3d_global_fun - u_REF_3d, u3d_global_fun - u_REF_3d) * dx)) / sqrt(assemble(inner(u_REF_3d, u_REF_3d) * dx))
        L2e_1D = sqrt(assemble(inner(u1d_new        - u_REF_1d, u1d_new - u_REF_1d) * dx))        / sqrt(assemble(inner(u_REF_1d, u_REF_1d) * dx))

        refL2_history_3D.append(L2e_3D)  # <-- relative L2 convergence to  REFERENCE solution
        refL2_history_1D.append(L2e_1D)  # <-- relative L2 convergence to  REFERENCE solution

        L2_REFNORM_history_3D.append( sqrt( assemble( inner( u_REF_3d, u_REF_3d ) * dx)) )        
        L2_REFNORM_history_1D.append( sqrt( assemble( inner( u_REF_1d, u_REF_1d ) * dx)) )        






        # relative L_inf error
        L_inf_3D = np.max( np.abs( u3d_global_fun.vector()[:] - u_REF_3d.vector()[:])) / np.max( np.abs( u_REF_3d.vector()[:]))
        L_inf_1D = np.max( np.abs( u1d_new.vector()[:]        - u_REF_1d.vector()[:])) / np.max( np.abs( u_REF_1d.vector()[:]))

        refL_inf_history_3D.append(L_inf_3D) # <-- relative L_inf convergence to  REFERENCE solution
        refL_inf_history_1D.append(L_inf_1D) # <-- relative L_inf convergence to  REFERENCE solution

        L_inf_REFNORM_history_3D.append(np.max( np.abs( u_REF_3d.vector()[:])))
        L_inf_REFNORM_history_1D.append(np.max( np.abs( u_REF_1d.vector()[:])))




        # relative H1 error                                                                                                                                                         
        H1e_3D = H1_norm(u3d_global_fun - u_REF_3d) / max(H1_norm(u_REF_3d), 1e-26)
        H1e_1D = H1_norm(u1d_new        - u_REF_1d) / max(H1_norm(u_REF_1d), 1e-26)
              
        refH1_history_3D.append(H1e_3D)  # <-- relative L2 convergence to  REFERENCE solution
        refH1_history_1D.append(H1e_1D)  # <-- relative L2 convergence to  REFERENCE solution

        H1_REFNORM_history_3D.append(max(H1_norm(u_REF_3d), 1e-26)) 
        H1_REFNORM_history_1D.append(max(H1_norm(u_REF_1d), 1e-26)) 


        '''
        # check jump L2 norm

        L2jump       = 0.
        scaledL2jump = 0.

        for i, V_loc in enumerate(locals_V3d):
            u_average_loc              = Function(V_loc)
            u_average_loc.vector()[:]  = El2g_dof_list[i].transpose().dot(global_3d_sol) # this is averaged on the boundary

            # boundary surface integral
            ds_loc = Measure('ds', domain = V_loc.mesh(), subdomain_data = submeshes3d_tags[i])
            # Virtual (DD) boundary surface integral tag
            virtual = 222

            # || u_i - u_average||^2_\Gamma_i ----deviation from the average
            L2jump_loc = assemble(inner(u3d_fun_new[i] - u_average_loc, 
                                        u3d_fun_new[i] - u_average_loc) * ds_loc( virtual ))

            # then sum and sqrt
            L2jump       +=  L2jump_loc

            scaledL2jump +=  (1./h) * L2jump_loc # h-scaling for H1 conformity

        L2jump = sqrt(L2jump)
        L2jump_history.append(L2jump)


        scaledL2jump = sqrt(scaledL2jump)
        scaled_L2jump_history.append(scaledL2jump)

        combined_norm = sqrt(H1_norm(u3d_global_fun - u_REF_3d)**2  + scaledL2jump**2 )/max(H1_norm(u_REF_3d), 1e-26)     

        # DG_norm=
        DG_norm_history.append(...)

        '''

        # ------------------------------------------------------------
        # DG-type relative error:
        #
        # ||e||_DG^2 =
        #   sum_i ||u_i - u_ref||^2_{H1(Omega_i)}
        # + sum_i h_F^{-1} ||u_i - u_average||^2_{L2(Gamma_i)}
        #
        # relative DG error = ||e||_DG / ||u_ref||_{H1(Omega)}
        # ------------------------------------------------------------
        
        brokenH1_sq   = 0.0
        L2jump_sq     = 0.0
        scaledjump_sq = 0.0
        
        virtual = 222
        
        for i, V_loc in enumerate(locals_V3d):
        
            mesh_loc = V_loc.mesh()
        
            dx_loc = Measure("dx", domain=mesh_loc)
            ds_loc = Measure("ds", domain=mesh_loc, subdomain_data=submeshes3d_tags[i])
        
            # local numerical solution u_i
            u_loc = u3d_fun_new[i]
        
            # restriction of reference solution to subdomain i
            u_ref_loc = Function(V_loc)
            u_ref_loc.vector()[:] = El2g_dof_list[i].transpose().dot(u_REF_3d.vector()[:])
        
            # restriction of averaged/global DD solution to subdomain i
            u_average_loc = Function(V_loc)
            u_average_loc.vector()[:] = El2g_dof_list[i].transpose().dot(global_3d_sol)
        
            # --------------------------------------------------------
            # Broken H1 contribution:
            # ||u_i - u_ref||^2_{H1(Omega_i)}
            # --------------------------------------------------------
            e_loc = u_loc - u_ref_loc
        
            brokenH1_sq += assemble(
                (inner(e_loc, e_loc) + inner(grad(e_loc), grad(e_loc))) * dx_loc
            )
        
            # --------------------------------------------------------
            # Unscaled jump contribution:
            # ||u_i - u_average||^2_{L2(Gamma_i)}
            # --------------------------------------------------------
            jump_loc = u_loc - u_average_loc
        
            L2jump_loc = assemble(
                inner(jump_loc, jump_loc) * ds_loc(virtual)
            )
        
            L2jump_sq += L2jump_loc
        
            # --------------------------------------------------------
            # Scaled jump contribution:
            # h_F^{-1} ||u_i - u_average||^2_{L2(Gamma_i)}
            #
            # CellDiameter(mesh_loc) gives a local FE length scale.
            # This is preferable to the global DD scale h = 1/n_local.
            # --------------------------------------------------------
        
            scaledjump_sq += assemble(
                (1.0 / h) * inner(jump_loc, jump_loc) * ds_loc(virtual)
            )
         
         
        # absolute components
        brokenH1     = sqrt(brokenH1_sq)
        L2jump       = sqrt(L2jump_sq)
        scaledL2jump = sqrt(scaledjump_sq)
        
        # reference H1 norm on the global conforming reference space
        H1_ref_3D = max(H1_norm(u_REF_3d), 1e-26)
        
        # relative components
        brokenH1_rel     = brokenH1 / H1_ref_3D
        scaledL2jump_rel = scaledL2jump / H1_ref_3D
        
        # DG-type relative norm
        DG_norm = sqrt(brokenH1_sq + scaledjump_sq) / H1_ref_3D
        
        # store histories
        L2jump_history.append(L2jump)
        scaled_L2jump_history.append(scaledL2jump)
        DG_norm_history.append(DG_norm)



        print("\n-----------------------------------\n")
        print('control criteria:', stagnation_3D)
        print('3D L2 relative error 3d:', L2e_3D)
        print('3D H1 relative error 3d:', H1e_3D)
        print('3D L inf relative error:', L_inf_3D)
        print('1D L2 relative error 3d:', L2e_1D)
        print('1D H1 relative error 3d:', H1e_1D)
        print('1D L inf relative  error:', L_inf_1D)
        print('L2jump:', L2jump)
        print('scaledL2jump:', scaledL2jump)
        print('brokenH1 relative error:', brokenH1_rel)
        print('DG_relative error:', DG_norm)

        if coarse: 
            print('FULL residual correction:',     corr_tot)
            print('INTERIOR residual correction',  corr_internal)
            print('INTERFACE residual correction', corr_interface)

        print('-----------------it-----------------', kk)
        print("\n-----------------------------------\n")

        # ABSOLUTE ERROR
        err_percentage             = Function(W[0])  # <---not a relative error!
        err_percentage.vector()[:] = np.abs(u3d_global_fun.vector()[:] - u_REF_3d.vector()[:] )




        # FOR SAVING------------------------------------------------
        if kk%args.save_every==0:
            u3d_global_fun.rename("u", "u")
            u1d_new.rename("u1d", "u1d")
            err_percentage.rename("e_ABS", "e_ABS")

            File(f'{out_folder}3d_sol_alpha_{alpha}_f1{flagrom1}_f2{flagrom2}_f3{flagrom3}_{kk}.pvd')                 << u3d_global_fun, kk
            File(f'{out_folder}1d_sol_alpha_{alpha}_f1{flagrom1}_f2{flagrom2}_f3{flagrom3}_{kk}.pvd')                 << u1d_new       , kk
            File(f'{out_folder}absolute_error_percent_alpha_{alpha}_f1{flagrom1}_f2{flagrom2}_f3{flagrom3}_{kk}.pvd') << err_percentage, kk
        #----------------------------------------------------------



    # CHECK STOPPING CRITERIA #

    # Append the change to the history
    window_stagnation.append(stagnation_3D)
    if len(window_stagnation) > stagnation_window:
        window_stagnation.pop(0)

    # Check stagnation condition
    if len(window_stagnation) == stagnation_window:

        if all(abs(old) > 1 for old in window_stagnation):
            print(f"Stagnation detected. Stopping at iteration {kk}.")
            break



    if BREAKLOOP == 1:
        break



    # CHECK CONVERGENCE CRITERIA #
    if stagnation_3D < conv_tol:
       print(f"CONVERGENCE. Stopping at iteration {kk}.")

       BREAKLOOP = 1


    
    # NEW STEP #
    kk = kk + 1
    # set new starting 3d solution
    u3d_old = u3d_global_fun

    # set new starting 1d solution
    u1d_old.vector()[:] = u1d_new.vector()[:].copy()

    # divide in 1d subsolutions
    if (kk - 1)%5 == 0:
        u1d_loc = assemble_local_fun1d(u1d_old, array_dict1d, mappa_indici1d, submeshes1d, emptyvoxellist)

        #project 1d solutions
        tottime = 0
        start = time.time()
        for i in range(len(submeshes3d)):
            if emptyvoxellist[i] == 1:
                 u1d_interp[i] = interpolate_1d_on_3d(u1d_loc[i], u1d_interp[i], locals_idx[i]) 
        tottime += (time.time() - start)
        print("Time elapsed for projection: " + str(tottime) + " seconds.")

    # END LOOP


end_SOLVE = time.time()
solve_time = end_SOLVE - start_SOLVE #<--TIME
print("\nSOLVER TIME:", solve_time, "\n")
print("\nTOTAL  TIME:", time.time() - start_total, "\n")

if montecarlo:
    for key in snapshot_data:
        for i, data_list in snapshot_data[key].items():
            mat = np.stack(data_list)
            np.save(os.path.join(out_base, f"{key}_sub{i}_g{graph_name}.npy"), mat)

N = len(refL2_history_3D)  # or any reliable reference

def safe_array(lst):
    if len(lst) == 0:
        return np.full(N, np.nan)
    return np.array(lst)



# print table with convergence data--------------------------------------------------------
vectors_dict = {
                 "Stagn3d"                      : safe_array( stagnation_history_3D),
                 "Stagn1d"                      : safe_array( stagnation_history_1D),
                 "L2e_3d"                       : safe_array( refL2_history_3D),
                 "L2e_1d"                       : safe_array( refL2_history_1D),
                 "H1e_3d"                       : safe_array( refH1_history_3D),
                 "H1e_1d"                       : safe_array( refH1_history_1D),
                 "L_inf_e_3d"                   : safe_array( refL_inf_history_3D),
                 "L_inf_e_1d"                   : safe_array( refL_inf_history_1D),
                 "L2_jump"                      : safe_array(L2jump_history),
                 "scaled2_jump"                 : safe_array(scaled_L2jump_history),
                 "DG_norm"                      : safe_array(DG_norm_history),
                 "corr_interior_history "       : safe_array( corr_internal_history),    
                 "corr_interface_history   "    : safe_array( corr_interface_history ), 
                 "corr_tot_history  "           : safe_array( corr_tot_history ),
                 "REF1D_L2"                     : safe_array( L2_REFNORM_history_1D),
                 "REF3D_L2"                     : safe_array( L2_REFNORM_history_3D),
                 "REF1D_L_inf"                  : safe_array( L_inf_REFNORM_history_1D),
                 "REF3D_L_inf"                  : safe_array( L_inf_REFNORM_history_3D),
                 "REF1D_H1"                     : safe_array( L2_REFNORM_history_1D),
                 "REF3D_H1"                     : safe_array( H1_REFNORM_history_3D)
              }

tabTOT, headers_string = build_comparison_table(vectors_dict)
# Create output directory if it doesn't exist
tab_output_directory = f'{out_folder}tabs/'
os.makedirs(tab_output_directory, exist_ok=True)

# Save the table to a file
tab_output_file = f'{tab_output_directory}tab_convergence_1D_{graph_name}_nref_{nref}_n_local_{n_local}_alpha_{alpha}_f1{flagrom1}_f2{flagrom2}_f3{flagrom3}.txt'
np.savetxt(tab_output_file, tabTOT, header=headers_string, comments='', fmt='%s')
#------------------------------------------------------------------------------------------
print("\n THAT'S ALL FOLKS")