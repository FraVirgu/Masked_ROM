from pyclbr import Function
import time
import os
import numpy as np
from scipy.sparse import csr_matrix, save_npz, lil_matrix
from dolfin import *
from xii import *
from xii.assembler.average_matrix import average_matrix as average_3d1d_matrix, trace_3d1d_matrix
from block.algebraic.hazmath import block_mat_to_block_dCSRmat
from petsc4py import PETSc
import haznics
from Boundary import Boundary,boundary
from dolfin import SubDomain

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
            f.parameters["flush_output"]          = True
            f.parameters["functions_share_mesh"]  = True
            self.u3d.rename("u3d", "3D pressure")
            f.write(self.u3d)

        # ── 1D solution ───────────────────────────────────────────────
        with XDMFFile(f"{output_folder}/u1d.xdmf") as f:
            f.parameters["flush_output"]          = True
            f.parameters["functions_share_mesh"]  = True
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

        print(f"ParaView files saved → {output_folder}/")
        print(f"  u3d.xdmf          — 3D pressure field")
        print(f"  u1d.xdmf          — 1D vessel pressure")
        print(f"  cell_markers.xdmf — inside/outside regions")
        print(f"  vessel_markers.xdmf — inlet/bulk/outlet tags")
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

        dx_     = Measure("dx", domain=self.meshQ)
        ds      = self.ds
        tag     = self.inlet_tag
        k3      = Constant(self.sigma3d)
        beta    = Constant(self.beta_nitsche)
        h_E     = MaxCellEdgeLength(self.meshQ)
        n_fct   = FacetNormal(self.meshQ)
        p_in    = Constant(1)
        u_out   = Constant(0.0)
        PENALTY = Constant(1e10)

        a = block_form(self.W, 2)
        m = block_form(self.W, 2)
        L = block_form(self.W, 1)

        # ── a[0][0]: 3D diffusion — identical in all branches ─────────────
        a[0][0] = (
            k3 * inner(grad(u), grad(v)) * self.d_omega(222)
            + k3 * inner(u, v)             * self.d_omega(222)
            + PENALTY * inner(u, v)        * self.d_omega(111)
        )

        # ── rhs — identical in all branches ───────────────────────────────
        L[0] = (
            inner(Constant(0), v) * self.d_omega(222)
            + PENALTY * inner(u_out, v) * self.d_omega(111)
        )
        L[1] = (
            - inner(p_in, dot(grad(q), n_fct)) * ds(tag, domain=self.meshQ)
            + beta * (h_E**-1) * inner(p_in, q) * ds(tag, domain=self.meshQ)
        )

        # ── Nitsche boundary terms — identical in all branches ────────────
        nitsche = (
            - inner(dot(grad(p), n_fct), q) * ds(tag, domain=self.meshQ)
            - inner(p, dot(grad(q), n_fct)) * ds(tag, domain=self.meshQ)
            + beta * (h_E**-1) * inner(p, q) * ds(tag, domain=self.meshQ)
        )

        # ══════════════════════════════════════════════════════════════════
        # BRANCH 1 — per-edge radii
        # ══════════════════════════════════════════════════════════════════
        if self.Q_radii is not None:
            n_V   = V.dofmap().global_dimension()
            n_Q   = Q.dofmap().global_dimension()
            C_mat = lil_matrix((n_Q, n_V))

            forms_00, forms_01, forms_10, forms_11, forms_a11 = [], [], [], [], []

            # one MeshFunction, unique tag per edge — same subdomain_data
            # object across all dx_i so UFL accepts sum(forms) ✓
            cell_marker = MeshFunction("size_t", self.meshQ, 1, 0)
            for i in range(self.meshQ.num_cells()):
                cell_marker[i] = i + 1

            dx_sub = Measure("dx", domain=self.meshQ, subdomain_data=cell_marker)

            for i in range(self.meshQ.num_cells()):
                v0, v1 = self.meshQ.cells()[i]
                Ri     = max(0.5 * (self.Q_radii[int(v0)] + self.Q_radii[int(v1)]), 0.005)

                gamma_i    = self.kappa * 2 * np.pi * Ri
                k1_i       = self.sigma1d_ref * np.pi * Ri**2
                cylinder_i = Circle(radius=Ri, degree=10)

                # same subdomain_data object, different tag ✓
                dx_i = dx_sub(i + 1)

                # C_mat: only rows for DOFs of edge i
                C_i_petsc             = average_3d1d_matrix(V, Q, cylinder_i)
                indptr, indices, data = C_i_petsc.getValuesCSR()
                C_i_full              = csr_matrix((data, indices, indptr),
                                                shape=(n_Q, n_V))
                for dof in Q.dofmap().cell_dofs(i):
                    C_mat[dof, :] = C_i_full[dof, :]

                Ru_i = Average(u, self.meshQ, cylinder_i)
                Rv_i = Average(v, self.meshQ, cylinder_i)

                forms_00.append( gamma_i * inner(Ru_i, Rv_i) * dx_i)
                forms_01.append(-gamma_i * inner(p,    Rv_i) * dx_i)
                forms_10.append(-gamma_i * inner(q,    Ru_i) * dx_i)
                forms_11.append( gamma_i * inner(p,    q)    * dx_i)
                forms_a11.append(k1_i   * inner(grad(p), grad(q)) * dx_i)

            self.C  = C_mat.tocsr()

            m[0][0] = sum(forms_00)
            m[0][1] = sum(forms_01)
            m[1][0] = sum(forms_10)
            m[1][1] = sum(forms_11)
            a[1][1] = sum(forms_a11) + nitsche

            self.AD, self.M, self.b = map(ii_assemble, (a, m, L))
            self.A = self.AD + self.M          # gamma baked into forms ✓

        # ══════════════════════════════════════════════════════════════════
        # BRANCH 2 — uniform cylinder average
        # ══════════════════════════════════════════════════════════════════
        elif self.coupling_radius > 0:
            cylinder = Circle(radius=self.coupling_radius, degree=10)
            Ru       = Average(u, self.meshQ, cylinder)
            Rv       = Average(v, self.meshQ, cylinder)
            C_petsc  = average_3d1d_matrix(V, Q, cylinder)

            m[0][0] =  inner(Ru, Rv) * dx_
            m[0][1] = -inner(p,  Rv) * dx_
            m[1][0] = -inner(q,  Ru) * dx_
            m[1][1] =  inner(p,  q)  * dx_
            a[1][1] = Constant(self.sigma1d) * inner(grad(p), grad(q)) * dx_ + nitsche
            self.C  = csr_matrix(C_petsc.getValuesCSR()[::-1], shape=C_petsc.size)

            self.AD, self.M, self.b = map(ii_assemble, (a, m, L))
            self.A = self.AD + self.gamma * self.M

        # ══════════════════════════════════════════════════════════════════
        # BRANCH 3 — pointwise trace (zero radius)
        # ══════════════════════════════════════════════════════════════════
        else:
            Ru      = Average(u, self.meshQ, None)
            Rv      = Average(v, self.meshQ, None)
            C_petsc = trace_3d1d_matrix(V, Q, self.meshQ)

            m[0][0] =  inner(Ru, Rv) * dx_
            m[0][1] = -inner(p,  Rv) * dx_
            m[1][0] = -inner(q,  Ru) * dx_
            m[1][1] =  inner(p,  q)  * dx_
            a[1][1] = Constant(self.sigma1d) * inner(grad(p), grad(q)) * dx_ + nitsche
            self.C  = csr_matrix(C_petsc.getValuesCSR()[::-1], shape=C_petsc.size)

            self.AD, self.M, self.b = map(ii_assemble, (a, m, L))
            self.A = self.AD + self.gamma * self.M

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