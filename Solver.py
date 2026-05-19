from pyclbr import Function
import time
import os
import numpy as np
from scipy.sparse import csr_matrix, save_npz, lil_matrix
from dolfin import *
from xii import *
from xii.assembler.average_matrix import average_matrix as average_3d1d_matrix, scalar_average_matrix, trace_3d1d_matrix
from block.algebraic.hazmath import block_mat_to_block_dCSRmat
from petsc4py import PETSc
import haznics
from Boundary import Boundary,boundary
from dolfin import SubDomain
from xii.linalg.matrix_utils import petsc_serial_matrix, is_number
import tqdm

def average_matrix_diff_radii(V, TV, TV_radii):
    '''
    Averaging matrix for reduction of g in V to TV by integration over shape.
    '''
    # We build a matrix representation of u in V -> Pi(u) in TV where
    #
    # Pi(u)(s) = |L(s)|^-1*\int_{L(s)}u(t) dx(s)
    #
    # Here L is the shape over which u is integrated for reduction.
    # Its measure is |L(s)|.
    mesh_x = TV.mesh().coordinates()
    # The idea for point evaluation/computing dofs of TV is to minimize
    # the number of evaluation. I mean a vector dof if done naively would
    # have to evaluate at same x number of component times.
    value_size = TV.ufl_element().value_size()
    
    mesh = V.mesh()
    # Eval at points will require serch
    tree = mesh.bounding_box_tree()
    limit = mesh.num_cells()

    TV_coordinates = TV.tabulate_dof_coordinates().reshape((TV.dim(), -1))
    line_mesh = TV.mesh()
    
    TV_dm = TV.dofmap()
    V_dm = V.dofmap()
    # For non scalar we plan to make compoenents by shift
    if value_size > 1:
        TV_dm = TV.sub(0).dofmap()

    Vel = V.element()               
    basis_values = np.zeros(V.element().space_dimension()*value_size)
    with petsc_serial_matrix(TV, V) as mat:

        for line_cell in tqdm.tqdm(cells(line_mesh), desc=f'Averaging over {line_mesh.num_cells()} cells',
                                   total=line_mesh.num_cells()):
            # Get the tangent (normal of the plane which cuts the virtual
            # surface to yield the bdry curve
            v0, v1 = mesh_x[line_cell.entities(0)]
            n = v0 - v1
            index_cell = line_cell.index()
            v0, v1 = TV.mesh().cells()[index_cell]
            Ri     = max(0.5 * (TV_radii[int(v0)] + TV_radii[int(v1)]), 0.005)
            shape = Circle(radius=Ri, degree=10)
            # The idea is now to minimize the point evaluation
            scalar_dofs = TV_dm.cell_dofs(line_cell.index())
            scalar_dofs_x = TV_coordinates[scalar_dofs]
            for scalar_row, avg_point in zip(scalar_dofs, scalar_dofs_x):
                # Avg point here has the role of 'height' coordinate
                quadrature = shape.quadrature(avg_point, n)
                integration_points = quadrature.points
                wq = quadrature.weights

                curve_measure = sum(wq)

                data = {}
                for index, ip in enumerate(integration_points):
                    c = tree.compute_first_entity_collision(Point(*ip))
                    if c >= limit:
                        c = None
                        continue

                    if c is None:
                        cs = tree.compute_entity_collisions(Point(*ip))[:1]
                    else:
                        cs = (c, )
                    # assert False
                    for c in cs:
                        Vcell = Cell(mesh, c)
                        vertex_coordinates = Vcell.get_vertex_coordinates()
                        cell_orientation = Vcell.orientation()
                        basis_values[:] = Vel.evaluate_basis_all(ip, vertex_coordinates, cell_orientation)

                        cols_ip = V_dm.cell_dofs(c)
                        values_ip = basis_values*wq[index]
                        # Add
                        for col, value in zip(cols_ip, values_ip.reshape((-1, value_size))):
                            if col in data:
                                data[col] += value/curve_measure
                            else:
                                data[col] = value/curve_measure
                            
                # The thing now that with data we can assign to several
                # rows of the matrix
                column_indices = np.array(list(data.keys()), dtype='int32')
                for shift in range(value_size):
                    row = scalar_row + shift
                    column_values = np.array([data[col][shift] for col in column_indices])
                    mat.setValues([row], column_indices, column_values, PETSc.InsertMode.INSERT_VALUES)
            # On to next avg point
        # On to next cell
    return mat





class FEniCSBoundaryWrapper(SubDomain):
    def __init__(self, boundary_obj):
        super().__init__()
        self._b = boundary_obj

    def inside(self, x, on_boundary):
        # on_boundary: True = outer skin of the BoxMesh
        # self._b(x):  True = inside your domain
        return on_boundary and self._b([x[0], x[1], x[2]])

class Solver3D1D:
    """
    Solves the LZ diffusion equation with:
      - Neumann BCs on the 3D domain
      - Nitsche (weak) Dirichlet BCs on the 1D network
    Coupled via average (or trace) operator.
    Preconditioned with metric AMG (HazNics).
    """

    def __init__(
        self,
        path_to_1D_mesh: str,
        n: int               = 10,
        max_radius: float    = 0.1,    # controls 3D box extension (fixed in original)
        coupling_radius: float = 0.0,  # controls averaging cylinder radius
        sigma3d: float       = 1e-3,
        sigma1d: float       = 1.0,
        kappa: float         = 1.0,
        beta_nitsche: float  = 5.0,
        inlet_tag: int       = 111,
        boundary: Boundary | None = None,
    ):
        # --- parameters ---
        self.path_to_1D_mesh  = path_to_1D_mesh
        self.n                = n
        self.max_radius       = max_radius                           # FIX 1: separate from coupling_radius
        self.coupling_radius  = coupling_radius
        self.sigma3d          = sigma3d
        self.sigma1d_ref = sigma1d          # raw, before R² scaling
        self.sigma1d     = sigma1d * np.pi * coupling_radius**2
        self.kappa            = kappa
        self.gamma            = kappa * 2 * np.pi * coupling_radius   # coupling strength
        self.beta_nitsche     = beta_nitsche
        self.inlet_tag        = inlet_tag
        self.boundary         = boundary
        # --- derived (populated by build() / solve()) ---
        self.meshV      = None   # 3D mesh
        self.meshQ      = None   # 1D mesh
        self.Q_markers  = None   # FIX 2: keep Q_markers alive on self
        self.V_markers = None 
        self.ds         = None   # boundary measure on 1D
        self.W          = None   # [V, Q] function spaces
        self.AD         = None   # diffusion block matrix
        self.M          = None   # coupling block matrix
        self.A          = None   # total matrix AD + gamma * M
        self.b          = None   # rhs
        self.C          = None   # coupling incidence matrix (CSR)
        self.V_DOF      = None   # number of 3D dofs

        self.x_np       = None   # solution as numpy array
        self.u3d        = None   # 3D solution Function
        self.u1d        = None   # 1D solution Function
        self.niters     = None   # solver iterations
        self.solve_time = None

    # =========================================================================
    # Public API
    # =========================================================================

    def build(self):
        """Load meshes and assemble the linear system. Returns self for chaining."""
        self._load_meshes()
        self._assemble_system()
        return self

    def solve(self):
        """Solve with HazNics metric AMG. Requires build() first."""
        self._require("A", "b", "W", "C", "AD", "M")

        # sanity check: non-zero rhs
        bb_norm = ii_convert(self.b).norm("l2")
        print(f"‖b‖ = {bb_norm:.6e}")
        if bb_norm == 0.0:
            raise RuntimeError(
                "RHS is zero — inlet markers (tag=111) were not found. "
                "Check that Q_markers is loaded correctly."
            )

        t0 = time.time()
        self.niters, _, self.x_np = self._solve_haznics(
            self.W, self.A, self.b, self.AD, self.M, self.C
        )
        self.solve_time = time.time() - t0
        self._split_solution()
        print(f"Solving elapsed time: {self.solve_time:.3f}s  |  iters: {self.niters}")
        return self

    def save(self, output_folder: str):
        """Save the numpy solution and print summary."""
        self._require("x_np", "u3d", "u1d")
        os.makedirs(output_folder, exist_ok=True)
        np.save(f"{output_folder}/solution.npy", self.x_np)
        print(f"Solution saved → {output_folder}/solution.npy")
        self._print_summary()

    def save_matrix(self, mat, path: str):
        """Save a matrix or vector to .npz (CSR format)."""
        if isinstance(mat, Vector):
            mat_np = np.array(mat)
        else:
            mat_np = ii_convert(mat).array()
        save_npz(path, csr_matrix(mat_np))

    def get_graph_distance(self):
        """
        Compute the distance from each 3D dof to the closest point on the 1D mesh.
        Returns a dolfin Function on V.
        """
        self._require("meshQ", "W")
        from closest_point_in_mesh import closest_point_in_mesh
        V    = self.W[0]
        dist = Function(V)
        for i, coord in enumerate(V.tabulate_dof_coordinates()):
            close_p = closest_point_in_mesh(coord, self.meshQ)
            dist.vector()[i] = np.linalg.norm(coord - close_p)
        return dist

    def save_paraview(self, output_folder: str):
        """Save 3D and 1D solutions + mesh markers for ParaView visualization."""
        self._require("u3d", "u1d", "meshV", "meshQ")
        os.makedirs(output_folder, exist_ok=True)

        # ── 3D solution ───────────────────────────────────────────────
        with XDMFFile(f"{output_folder}/u3d.xdmf") as f:
            f.parameters["flush_output"]         = True
            f.parameters["functions_share_mesh"] = True
            self.u3d.rename("u3d", "3D pressure")
            f.write(self.u3d)

        # ── 1D solution ───────────────────────────────────────────────
        with XDMFFile(f"{output_folder}/u1d.xdmf") as f:
            f.parameters["flush_output"]         = True
            f.parameters["functions_share_mesh"] = True
            self.u1d.rename("u1d", "1D vessel pressure")
            f.write(self.u1d)

        # ── 3D cell markers (inside=222 / outside=111) ────────────────
        with XDMFFile(f"{output_folder}/cell_markers.xdmf") as f:
            f.parameters["flush_output"] = True
            f.write(self.V_cell_markers)

        # ── 1D vertex markers (inlet=111 / bulk=555 / outlet=999) ─────
        with XDMFFile(f"{output_folder}/vessel_markers.xdmf") as f:
            f.parameters["flush_output"] = True
            f.write(self.Q_markers)

        # ── per-edge radius (dim=1 cell function on 1D mesh) ──────────
        if self.Q_radii is not None:
            # Q_radii is dim=0 (vertex). Build a dim=1 edge version
            # by averaging the two endpoint values — natural for ParaView
            Q          = self.W[1]
            edge_radii = MeshFunction("double", self.meshQ, 1, 0.0)
            for cell in cells(self.meshQ):
                v0, v1 = cell.entities(0)
                edge_radii[cell.index()] = 0.5 * (
                    self.Q_radii[int(v0)] + self.Q_radii[int(v1)]
                )
            with XDMFFile(f"{output_folder}/vessel_radii.xdmf") as f:
                f.parameters["flush_output"] = True
                edge_radii.rename("radius", "vessel radius")
                f.write(edge_radii)

            # also write as a 1D CG1 Function so ParaView can interpolate
            # smoothly along each vessel segment
            radius_fn = Function(Q)
            for v_idx in range(self.meshQ.num_vertices()):
                radius_fn.vector()[v_idx] = self.Q_radii[v_idx]
            with XDMFFile(f"{output_folder}/vessel_radii_smooth.xdmf") as f:
                f.parameters["flush_output"]         = True
                f.parameters["functions_share_mesh"] = True
                radius_fn.rename("radius_smooth", "vessel radius (smooth)")
                f.write(radius_fn)

        print(f"ParaView files saved → {output_folder}/")
        print(f"  u3d.xdmf                — 3D pressure field")
        print(f"  u1d.xdmf                — 1D vessel pressure")
        print(f"  cell_markers.xdmf       — inside/outside regions")
        print(f"  vessel_markers.xdmf     — inlet/bulk/outlet tags")
        if self.Q_radii is not None:
            print(f"  vessel_radii.xdmf       — per-edge radius (cell function)")
            print(f"  vessel_radii_smooth.xdmf — per-vertex radius (CG1 function)")
    # =========================================================================
    # Private build steps
    # =========================================================================

    def _load_meshes(self):
        """Build the 3D BoxMesh and load the 1D network mesh + markers."""

        # FIX 1: 3D box uses max_radius (= 0.1 in original), not coupling_radius
        r          = self.max_radius
        inf_pt     = Point(-1 - 1.1*r, -1 - 1.1*r, -1 - 1.1*r)
        max_pt     = Point( 1 + 1.1*r,  1 + 1.1*r,  1 + 1.1*r)
        self.boundary = boundary
        self.meshV = BoxMesh(inf_pt, max_pt, self.n, self.n, self.n)
        
        

        # MARKET dim=3 => cells (tetrahedra)       
        # Seperate the cell inside the subdomain (anatomical domain) from the cell outside the subdomain (outside anatomical domain)
        
        self.V_cell_markers = MeshFunction("size_t", self.meshV, 3, 222)
        for cell in cells(self.meshV):
            mp = cell.midpoint()
            point = [mp.x(), mp.y(), mp.z()]
            if self.boundary is not None and self.boundary(point):
                self.V_cell_markers[cell] = 222   # inside anatomical domain
            else:
                self.V_cell_markers[cell] = 111   # outside anatomical domain
        
        self.d_omega = Measure("dx", domain=self.meshV,
                                            subdomain_data=self.V_cell_markers)

        
        # 1: 1D mesh
        self.meshQ = Mesh()
        with XDMFFile(f"{self.path_to_1D_mesh}marked_mesh.xdmf") as f:
            f.read(self.meshQ)

        # 2: Q_markers 
        self.Q_markers = MeshFunction("size_t", self.meshQ, 0)
        xdmf_markers   = XDMFFile(f"{self.path_to_1D_mesh}markers.xdmf")
        xdmf_markers.read(self.Q_markers)
        xdmf_markers.close()

        # 3: radii  (vertex function, dim=0)
        radii_path = f"{self.path_to_1D_mesh}radii.xdmf"
        if os.path.exists(radii_path):
            self.Q_radii = MeshFunction("double", self.meshQ, 0)
            xdmf_radii   = XDMFFile(radii_path)
            xdmf_radii.read(self.Q_radii)
            xdmf_radii.close()
            r_arr = self.Q_radii.array()
            print(f"Q_radii loaded: min={r_arr.min():.4f}  max={r_arr.max():.4f}")
        else:
            self.Q_radii = None
            print("No radii file found — using uniform coupling_radius")

        self.ds = Measure("ds", domain=self.meshQ)
        self.ds = self.ds(subdomain_data=self.Q_markers)

        print(f"3D mesh: {self.meshV.num_cells()} cells  |  "
              f"1D mesh: {self.meshQ.num_cells()} cells")

    def _assemble_system(self):
        """Assemble stiffness, coupling, and rhs blocks."""
        self._require("meshV", "meshQ", "ds")

        V          = FunctionSpace(self.meshV, "CG", 1)
        Q          = FunctionSpace(self.meshQ, "CG", 1)
        self.W     = [V, Q]
        self.V_DOF = V.dofmap().global_dimension()
        print(f"3D DOFs: {self.V_DOF}")

        u, v = TrialFunction(V), TestFunction(V)
        p, q = TrialFunction(Q), TestFunction(Q)

        ds      = self.ds
        tag     = self.inlet_tag
        k3      = Constant(self.sigma3d)
        beta    = Constant(self.beta_nitsche)
        h_E     = MaxCellEdgeLength(self.meshQ)
        n_fct   = FacetNormal(self.meshQ)
        p_in    = Constant(1)
        u_out   = Constant(0.0)
        PENALTY = Constant(1e10)
        dx_     = Measure("dx", domain=self.meshQ)

        if self.Q_radii is None:
            if self.coupling_radius <= 0:
                raise ValueError("Either Q_radii or coupling_radius > 0 must be provided.")
            self.Q_radii = MeshFunction("double", self.meshQ, 0, self.coupling_radius)
            print(f"Q_radii not found — using uniform radius {self.coupling_radius}")

        n_V = V.dofmap().global_dimension()
        n_Q = Q.dofmap().global_dimension()

        # ── C: single pass, variable radius ───────────────────────────────────
        C_petsc            = average_matrix_diff_radii(V, Q, self.Q_radii)
        indptr, idx, data_ = C_petsc.getValuesCSR()
        C                  = csr_matrix((data_, idx, indptr), shape=(n_Q, n_V))
        self.C             = C

        # ── DG0 coefficients: one value per edge, no assembler ────────────────
        DG0     = FunctionSpace(self.meshQ, "DG", 0)
        gamma_f = Function(DG0)
        k1_f    = Function(DG0)
        for i in range(self.meshQ.num_cells()):
            cv0, cv1            = self.meshQ.cells()[i]
            Ri                  = max(0.5*(self.Q_radii[int(cv0)] + self.Q_radii[int(cv1)]), 0.005)
            gamma_f.vector()[i] = self.kappa * 2 * np.pi * Ri
            k1_f.vector()[i]    = self.sigma1d_ref * np.pi * Ri**2

        # ── G: weighted 1D mass matrix — ONE assemble call ────────────────────
        G_dolfin      = assemble(gamma_f * inner(p, q) * dx_)
        gi, gj, gv    = as_backend_type(G_dolfin).mat().getValuesCSR()
        G             = csr_matrix((gv, gj, gi), shape=(n_Q, n_Q))

        # ── m blocks: pure scipy, zero FFC/averaging calls ────────────────────
        M_00 =  C.T @ G @ C    # (n_V × n_V)
        M_01 = -C.T @ G        # (n_V × n_Q)
        M_10 = -G @ C          # (n_Q × n_V)
        M_11 =  G              # (n_Q × n_Q)

        def to_dolfin(A_sp):
            """scipy CSR → dolfin PETSc matrix"""
            A_sp = csr_matrix(A_sp)
            pet  = PETSc.Mat().createAIJ(
                size = A_sp.shape,
                csr  = (A_sp.indptr.astype('int32'),
                        A_sp.indices.astype('int32'),
                        A_sp.data.copy())
            )
            pet.assemble()
            return PETScMatrix(pet)

        from block import block_mat as bmat
        self.M = bmat([[to_dolfin(M_00), to_dolfin(M_01)],
                    [to_dolfin(M_10), to_dolfin(M_11)]])

        # ── block forms: a and L only, no m ───────────────────────────────────
        a = block_form(self.W, 2)
        L = block_form(self.W, 1)

        a[0][0] = (
            k3 * inner(grad(u), grad(v)) * self.d_omega(222)
            + k3 * inner(u, v)             * self.d_omega(222)
            + PENALTY * inner(u, v)        * self.d_omega(111)
        )
        # k1_f is DG0 — single FFC call, no per-edge loop ✓
        a[1][1] = k1_f * inner(grad(p), grad(q)) * dx_ + (
            - inner(dot(grad(p), n_fct), q) * ds(tag, domain=self.meshQ)
            - inner(p, dot(grad(q), n_fct)) * ds(tag, domain=self.meshQ)
            + beta * (h_E**-1) * inner(p, q) * ds(tag, domain=self.meshQ)
        )

        L[0] = (
            inner(Constant(0), v) * self.d_omega(222)
            + PENALTY * inner(u_out, v) * self.d_omega(111)
        )
        L[1] = (
            - inner(p_in, dot(grad(q), n_fct)) * ds(tag, domain=self.meshQ)
            + beta * (h_E**-1) * inner(p_in, q) * ds(tag, domain=self.meshQ)
        )

        self.AD = ii_assemble(a)
        self.b  = ii_assemble(L)
        self.A  = self.AD + self.M

        print("System assembled.")
    def _split_solution(self):
        """Split the flat numpy solution into 3D and 1D dolfin Functions."""
        dimV        = self.W[0].dim()
        self.u3d    = Function(self.W[0])
        self.u1d    = Function(self.W[1])
        self.u3d.vector()[:] = self.x_np[:dimV]
        self.u1d.vector()[:] = self.x_np[dimV:]

    # =========================================================================
    # HazNics solver (static — mirrors original solve_haznics exactly)
    # =========================================================================

    @staticmethod
    def _solve_haznics(W, A, b, AD, M, C):
        """Solve the block system with metric AMG (HazNics)."""

        def block_to_haz(AA):
            if hasattr(AA, "block_collapse"):
                AA = AA.block_collapse()
            brow, bcol = AA.blocks.shape
            for i in range(brow):
                for j in range(bcol):
                    AA[i][j] = ii_collapse(AA[i][j])
            return block_mat_to_block_dCSRmat(AA)

        dimW = sum(VV.dim() for VV in W)

        bb   = ii_convert(b)
        b_np = bb[:]
        bhaz = haznics.create_dvector(b_np)
        xhaz = haznics.dvec_create_p(dimW)

        Ahaz  = block_to_haz(A)
        Mhaz  = block_to_haz(M)
        ADhaz = block_to_haz(AD)

        # preserve original argument order: (data, indices, indptr)
        csr0, csr1, csr2 = C.indptr, C.indices, C.data
        Chaz = haznics.create_matrix(csr2, csr1, csr0, C.shape[1])

        niters = haznics.fenics_metric_amg_solver(Ahaz, bhaz, xhaz, ADhaz, Mhaz, Chaz)

        haznics.dvec_write("/tmp/solution_raw.dat", xhaz)
        x_np = np.loadtxt("/tmp/solution_raw.dat", skiprows=1)

        return niters, xhaz, x_np

    # =========================================================================
    # Utilities
    # =========================================================================

    def _require(self, *attrs):
        for attr in attrs:
            if getattr(self, attr) is None:
                raise RuntimeError(
                    f"'{attr}' is not available. "
                    f"Call build() first (or the relevant _build_* step)."
                )

    def _print_summary(self):
        dimV, dimQ = self.W[0].dim(), self.W[1].dim()
        print("=" * 60)
        print(f"sigma3d={self.sigma3d}, sigma1d={self.sigma1d:.6f}, "
              f"radius={self.coupling_radius}, kappa={self.kappa}")
        print(f"dim(V)={dimV}  dim(Q)={dimQ}  "
              f"hmax(V)={self.W[0].mesh().hmax():.2f}  "
              f"hmin(V)={self.W[0].mesh().hmin():.2f}  "
              f"hmin(Q)={self.W[1].mesh().hmin():.4f}  "
              f"niters={self.niters}")
        print("=" * 60)

    def __repr__(self):
        return (
            f"Solver3D1D(n={self.n}, max_radius={self.max_radius}, "
            f"coupling_radius={self.coupling_radius}, "
            f"sigma3d={self.sigma3d}, sigma1d={self.sigma1d:.6f}, "
            f"kappa={self.kappa}, gamma={self.gamma:.4f}, "
            f"built={self.W is not None})"
        )


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-mesh",  type=int,   required=True, help="mesh index")
    parser.add_argument("-rad",   type=float, required=True, help="coupling radius")
    parser.add_argument("-nnn",   type=int,   required=True, help="3D mesh size")
    parser.add_argument("-which", type=str,   required=True, help="train or test")
    args, _ = parser.parse_known_args()

    for arg, val in vars(args).items():
        if val is None:
            raise ValueError(f"Missing required argument: -{arg}")

   

    solver = Solver3D1D(
        path_to_1D_mesh = f"./nets/{args.which}/net01__",
        n               = args.nnn,
        max_radius      = 0.1,          # matches original fixed value
        coupling_radius = args.rad,
        sigma3d         = 1e-3,
        sigma1d         = 1.0,
        kappa           = 1.0,
    ).build().solve()

    solver.save(f"./solution/mesh{args.mesh}_rad{args.rad}_n{args.nnn}")
    solver.save_paraview(f"./solution/mesh{args.mesh}_rad{args.rad}_n{args.nnn}/paraview")