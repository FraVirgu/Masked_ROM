import time
import os
import numpy as np
from scipy.sparse import csr_matrix, save_npz
from dolfin import *
from xii import *
from block.algebraic.hazmath import block_mat_to_block_dCSRmat
from petsc4py import PETSc
import haznics
from Boundary import Boundary
from xii.linalg.matrix_utils import petsc_serial_matrix
import tqdm


# =============================================================================
# Averaging matrix with per-vertex radii
# =============================================================================

def average_matrix_diff_radii(V, TV, TV_radii):
    """
    Averaging matrix for reduction of g in V to TV by integration over
    a circle of radius R(s) at each point s of the 1D mesh.
    """
    mesh_x         = TV.mesh().coordinates()
    value_size     = TV.ufl_element().value_size()
    mesh           = V.mesh()
    tree           = mesh.bounding_box_tree()
    limit          = mesh.num_cells()
    TV_coordinates = TV.tabulate_dof_coordinates().reshape((TV.dim(), -1))
    line_mesh      = TV.mesh()
    TV_dm          = TV.dofmap()
    V_dm           = V.dofmap()

    if value_size > 1:
        TV_dm = TV.sub(0).dofmap()

    Vel          = V.element()
    basis_values = np.zeros(V.element().space_dimension() * value_size)

    with petsc_serial_matrix(TV, V) as mat:
        for line_cell in tqdm.tqdm(
            cells(line_mesh),
            desc=f"Averaging over {line_mesh.num_cells()} cells",
            total=line_mesh.num_cells(),
        ):
            v0, v1  = mesh_x[line_cell.entities(0)]
            n       = v0 - v1
            idx_c   = line_cell.index()
            lv0, lv1 = TV.mesh().cells()[idx_c]
            Ri      = max(0.5 * (TV_radii[int(lv0)] + TV_radii[int(lv1)]), 0.005)
            shape   = Circle(radius=Ri, degree=10)

            scalar_dofs   = TV_dm.cell_dofs(idx_c)
            scalar_dofs_x = TV_coordinates[scalar_dofs]

            for scalar_row, avg_point in zip(scalar_dofs, scalar_dofs_x):
                quadrature         = shape.quadrature(avg_point, n)
                integration_points = quadrature.points
                wq                 = quadrature.weights
                curve_measure      = sum(wq)
                data               = {}

                for index, ip in enumerate(integration_points):
                    c = tree.compute_first_entity_collision(Point(*ip))
                    if c >= limit:
                        continue
                    for c in (c,):
                        Vcell              = Cell(mesh, c)
                        vertex_coordinates = Vcell.get_vertex_coordinates()
                        cell_orientation   = Vcell.orientation()
                        basis_values[:]    = Vel.evaluate_basis_all(
                            ip, vertex_coordinates, cell_orientation
                        )
                        cols_ip   = V_dm.cell_dofs(c)
                        values_ip = basis_values * wq[index]
                        for col, value in zip(
                            cols_ip, values_ip.reshape((-1, value_size))
                        ):
                            if col in data:
                                data[col] += value / curve_measure
                            else:
                                data[col]  = value / curve_measure

                column_indices = np.array(list(data.keys()), dtype="int32")
                for shift in range(value_size):
                    row           = scalar_row + shift
                    column_values = np.array(
                        [data[col][shift] for col in column_indices]
                    )
                    mat.setValues(
                        [row], column_indices, column_values,
                        PETSc.InsertMode.INSERT_VALUES,
                    )
    return mat


# =============================================================================
# Solver3D1D
# =============================================================================

class Solver3D1D:
    """
    Solves the 3D-1D coupled oxygen perfusion equation:

        3D:  -∇·(σ₃D ∇u) + σ₃D·u = γ(u_V - u_t)·δ_Λ   in Ω
        1D:  -d/ds(σ₁D·πR²·du_V/ds) = γ(u_t - u_V)      in Λ

    Requires:
        - boundary  : Boundary object defining the anatomical domain
        - radii     : loaded from {path_to_1D_mesh}radii.xdmf (per-vertex)
        - markers   : loaded from {path_to_1D_mesh}markers.xdmf (111/555/999)

    Will raise if boundary or radii are missing.
    """

    def __init__(
        self,
        path_to_1D_mesh : str,
        boundary        : Boundary,      # REQUIRED — no default
        n               : int   = 10,
        sigma3d         : float = 1e-3,
        sigma1d         : float = 1.0,
        kappa           : float = 1.0,
        beta_nitsche    : float = 5.0,
        inlet_tag       : int   = 111,
        exterior        : str   = "dirichlet",
        penalty         : float = 1.0,
    ):
        if boundary is None:
            raise ValueError(
                "boundary is required — pass a Boundary object built from "
                "boundary_from_obj() or an analytic function."
            )
        if exterior not in ("penalty", "restrict", "dirichlet"):
            raise ValueError(
                f"exterior must be 'penalty', 'restrict' or 'dirichlet', "
                f"got {exterior!r}."
            )

        # --- parameters ---
        self.path_to_1D_mesh = path_to_1D_mesh
        self.boundary        = boundary
        self.n               = n
        self.sigma3d         = sigma3d
        self.sigma1d_ref     = sigma1d
        self.kappa           = kappa
        self.beta_nitsche    = beta_nitsche
        self.inlet_tag       = inlet_tag

        # How the exterior (111) region of the background box is handled:
        #
        # 'restrict'  : V lives on the SubMesh of the 222 cells. The exterior
        #               DOFs do not exist. No parameter. This is the reference.
        # 'dirichlet' : V lives on the full box, but the DOFs with no support in
        #               any 222 cell are eliminated exactly (zero row/column, 1
        #               on the diagonal, 0 on the rhs). A DOF pinned to zero
        #               contributes nothing to any interior equation, so this
        #               reproduces 'restrict' to solver tolerance — but keeps
        #               u3d defined on the full box mesh.
        # 'penalty'   : V on the full box, exterior pinned by
        #               penalty*inner(u,v)*dx(111). Only an APPROXIMATION: the
        #               integral also hits the interface vertices shared with
        #               the 222 cells, so a penalty large enough to pin the
        #               exterior also contaminates the interior. Kept for
        #               comparison; prefer 'restrict' or 'dirichlet'.
        self.exterior = exterior
        self.penalty  = penalty
        if exterior == "penalty" and penalty <= 0.0:
            raise ValueError(
                "penalty must be > 0 with exterior='penalty': the exterior DOFs "
                "appear in no other term, so penalty=0 leaves them with an "
                "all-zero row/column and the matrix is singular (UMFPACK -5). "
                "Use exterior='restrict' to remove those DOFs instead."
            )

        # --- populated by build() ---
        self.meshV          = None
        self.meshQ          = None
        self.Q_markers      = None
        self.Q_radii        = None   # loaded from xdmf — REQUIRED
        self.V_cell_markers = None
        self.d_omega        = None
        self.ds             = None
        self.meshV_full     = None   # background box (== meshV when exterior='penalty')
        self.parent_vertex  = None   # submesh vertex -> box vertex (exterior='restrict')
        self.ext_dofs       = None   # eliminated V dofs      (exterior='dirichlet')
        self.int_dofs       = None   # kept V dofs            (exterior='dirichlet')
        self.W              = None
        self.AD             = None
        self.M              = None
        self.A              = None
        self.b              = None
        self.C              = None
        self.V_DOF          = None

        # --- populated by solve() ---
        self.x_np       = None
        self.u3d        = None
        self.u1d        = None
        self.niters     = None
        self.solve_time = None

    # =========================================================================
    # Public API
    # =========================================================================

    def build(self):
        """Load meshes and assemble the linear system. Returns self."""
        self._load_meshes()
        self._assemble_system()
        return self

    def solve(self):
        """Solve with HazNics metric AMG. Requires build() first."""
        self._require("A", "b", "W", "C", "AD", "M")

        bb_norm = ii_convert(self.b).norm("l2")
        print(f"‖b‖ = {bb_norm:.6e}")
        if bb_norm == 0.0:
            raise RuntimeError(
                "RHS is zero — inlet marker (tag=111) not found. "
                "Check markers.xdmf."
            )

        t0 = time.time()
        self.niters, _, self.x_np = self._solve_haznics(
            self.W, self.A, self.b, self.AD, self.M, self.C
        )
        self.solve_time = time.time() - t0
        self._split_solution()
        print(f"Solved in {self.solve_time:.3f}s  |  iters: {self.niters}")
        return self

    def save(self, output_folder: str):
        """Save numpy solution array."""
        self._require("x_np", "u3d", "u1d")
        os.makedirs(output_folder, exist_ok=True)
        np.save(f"{output_folder}/solution.npy", self.x_np)
        print(f"Solution saved → {output_folder}/solution.npy")
        self._print_summary()

    def save_paraview(self, output_folder: str):
        """Save all fields for ParaView visualization."""
        self._require("u3d", "u1d", "meshV", "meshQ")
        os.makedirs(output_folder, exist_ok=True)

        # 3D pressure
        with XDMFFile(f"{output_folder}/u3d.xdmf") as f:
            f.parameters["flush_output"]         = True
            f.parameters["functions_share_mesh"] = True
            self.u3d.rename("u3d", "3D pressure")
            f.write(self.u3d)

        # 1D vessel pressure
        with XDMFFile(f"{output_folder}/u1d.xdmf") as f:
            f.parameters["flush_output"]         = True
            f.parameters["functions_share_mesh"] = True
            self.u1d.rename("u1d", "1D vessel pressure")
            f.write(self.u1d)

        # 3D cell markers (222=inside / 111=outside)
        with XDMFFile(f"{output_folder}/cell_markers.xdmf") as f:
            f.parameters["flush_output"] = True
            f.write(self.V_cell_markers)

        # 1D vertex markers (111=inlet / 555=interior / 999=outlet)
        with XDMFFile(f"{output_folder}/vessel_markers.xdmf") as f:
            f.parameters["flush_output"] = True
            f.write(self.Q_markers)

        # per-edge radius (cell function, averaged from vertex radii)
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

        # per-vertex radius (CG1, smooth interpolation in ParaView)
        radius_fn = Function(Q)
        for v_idx in range(self.meshQ.num_vertices()):
            radius_fn.vector()[v_idx] = self.Q_radii[v_idx]
        with XDMFFile(f"{output_folder}/vessel_radii_smooth.xdmf") as f:
            f.parameters["flush_output"]         = True
            f.parameters["functions_share_mesh"] = True
            radius_fn.rename("radius_smooth", "vessel radius (smooth)")
            f.write(radius_fn)

        print(f"ParaView files → {output_folder}/")
        for fname in [
            "u3d.xdmf", "u1d.xdmf", "cell_markers.xdmf",
            "vessel_markers.xdmf", "vessel_radii.xdmf", "vessel_radii_smooth.xdmf",
        ]:
            print(f"  {fname}")

    # =========================================================================
    # Private build steps
    # =========================================================================

    def _load_meshes(self):
        """Build 3D BoxMesh, mark anatomical domain, load 1D mesh + radii."""

        # --- 3D mesh: fixed [-1,1]^3 ---
        self.meshV = BoxMesh(Point(-1, -1, -1), Point(1, 1, 1),
                             self.n, self.n, self.n)

        # --- mark cells: 222=inside domain / 111=outside ---
        self.V_cell_markers = MeshFunction("size_t", self.meshV, 3, 111)
        inside_count = 0
        for cell in cells(self.meshV):
            mp = cell.midpoint()
            if self.boundary([mp.x(), mp.y(), mp.z()]):
                self.V_cell_markers[cell] = 222
                inside_count += 1

        total = self.meshV.num_cells()
        print(f"3D mesh: {total} cells  |  "
              f"inside domain: {inside_count} ({inside_count/total*100:.1f}%)")
        if inside_count == 0:
            raise RuntimeError(
                "No cell midpoint falls inside the boundary — the interior is "
                "empty. Check the boundary scale/center."
            )

        self.meshV_full = self.meshV

        if self.exterior == "restrict":
            # Keep only the cells marked 222. The exterior DOFs then simply do
            # not exist, so no penalty is needed to make a[0][0] nonsingular.
            submesh            = SubMesh(self.meshV, self.V_cell_markers, 222)
            self.parent_vertex = submesh.data().array(
                "parent_vertex_indices", 0
            ).copy()
            self.meshV         = submesh

            # Every cell of the submesh is interior: re-mark uniformly as 222 so
            # the d_omega(222) forms below carry over unchanged.
            self.V_cell_markers = MeshFunction("size_t", self.meshV, 3, 222)
            print(f"3D mesh restricted to interior: {self.meshV.num_cells()} cells "
                  f"| {self.meshV.num_vertices()} vertices")

        self.d_omega = Measure("dx", domain=self.meshV,
                               subdomain_data=self.V_cell_markers)

        # --- 1D mesh ---
        self.meshQ = Mesh()
        with XDMFFile(f"{self.path_to_1D_mesh}marked_mesh.xdmf") as f:
            f.read(self.meshQ)

        # --- markers (111=inlet / 555=interior / 999=outlet) ---
        self.Q_markers = MeshFunction("size_t", self.meshQ, 0)
        xdmf_m = XDMFFile(f"{self.path_to_1D_mesh}markers.xdmf")
        xdmf_m.read(self.Q_markers)
        xdmf_m.close()

        # --- radii: REQUIRED ---
        radii_path = f"{self.path_to_1D_mesh}radii.xdmf"
        if not os.path.exists(radii_path):
            raise RuntimeError(
                f"radii.xdmf not found at '{radii_path}'. "
                f"Run CCOVascularMesh.export_xdmf() first."
            )
        self.Q_radii = MeshFunction("double", self.meshQ, 0)
        xdmf_r = XDMFFile(radii_path)
        xdmf_r.read(self.Q_radii)
        xdmf_r.close()
        r_arr = self.Q_radii.array()
        if (r_arr <= 0).any():
            raise RuntimeError(
                f"radii.xdmf contains {(r_arr <= 0).sum()} non-positive radii."
            )
        print(f"Q_radii: min={r_arr.min():.6f}  max={r_arr.max():.6f}")

        self.ds = Measure("ds", domain=self.meshQ,
                          subdomain_data=self.Q_markers)

        print(f"1D mesh: {self.meshQ.num_cells()} edges  |  "
              f"{self.meshQ.num_vertices()} vertices")

    def _assemble_system(self):
        """Assemble stiffness, coupling, and rhs blocks."""
        self._require("meshV", "meshQ", "ds", "Q_radii")

        V          = FunctionSpace(self.meshV, "CG", 1)
        Q          = FunctionSpace(self.meshQ, "CG", 1)
        self.W     = [V, Q]
        self.V_DOF = V.dofmap().global_dimension()
        print(f"3D DOFs: {self.V_DOF}  |  1D DOFs: {Q.dofmap().global_dimension()}")

        if self.exterior == "dirichlet":
            self._find_exterior_dofs(V)

        u, v   = TrialFunction(V), TestFunction(V)
        p, q   = TrialFunction(Q), TestFunction(Q)
        ds     = self.ds
        tag    = self.inlet_tag
        k3     = Constant(self.sigma3d)
        beta   = Constant(self.beta_nitsche)
        h_E    = MaxCellEdgeLength(self.meshQ)
        n_fct  = FacetNormal(self.meshQ)
        p_in   = Constant(1.0)
        u_out  = Constant(0.0)
        dx_    = Measure("dx", domain=self.meshQ)

        n_V = V.dofmap().global_dimension()
        n_Q = Q.dofmap().global_dimension()

        # --- averaging matrix C (variable radius per edge) ---
        C_petsc            = average_matrix_diff_radii(V, Q, self.Q_radii)
        indptr, idx, data_ = C_petsc.getValuesCSR()
        C                  = csr_matrix((data_, idx, indptr), shape=(n_Q, n_V))

        if self.exterior == "dirichlet":
            # A coupling circle near the interface can sample an exterior cell,
            # so C may have columns on eliminated DOFs. Those DOFs are 0, so
            # their contribution to the coupling is 0 — drop the columns before
            # M is formed, otherwise M would reintroduce entries on rows that
            # the elimination below zeroes out.
            is_ext = np.zeros(n_V, dtype=bool)
            is_ext[self.ext_dofs] = True

            Ccoo = C.tocoo()
            drop = is_ext[Ccoo.col]
            lost = np.abs(Ccoo.data[drop]).sum()

            C = csr_matrix(
                (Ccoo.data[~drop], (Ccoo.row[~drop], Ccoo.col[~drop])),
                shape=C.shape,
            )
            C.eliminate_zeros()
            print(f"C: zeroed {len(self.ext_dofs)} exterior columns "
                  f"(dropped weight {lost:.3e})")

        self.C = C

        # --- per-edge DG0 coefficients from per-vertex radii ---
        DG0     = FunctionSpace(self.meshQ, "DG", 0)
        gamma_f = Function(DG0)
        k1_f    = Function(DG0)
        for i in range(self.meshQ.num_cells()):
            cv0, cv1            = self.meshQ.cells()[i]
            Ri                  = max(
                0.5 * (self.Q_radii[int(cv0)] + self.Q_radii[int(cv1)]), 0.005
            )
            gamma_f.vector()[i] = self.kappa * 2 * np.pi * Ri
            k1_f.vector()[i]    = self.sigma1d_ref * np.pi * Ri ** 2

        # --- coupling mass matrix G ---
        G_dolfin   = assemble(gamma_f * inner(p, q) * dx_)
        gi, gj, gv = as_backend_type(G_dolfin).mat().getValuesCSR()
        G          = csr_matrix((gv, gj, gi), shape=(n_Q, n_Q))

        # --- coupling blocks ---
        M_00 =  C.T @ G @ C
        M_01 = -C.T @ G
        M_10 = -G @ C
        M_11 =  G

        def to_dolfin(A_sp):
            A_sp = csr_matrix(A_sp)
            pet  = PETSc.Mat().createAIJ(
                size=A_sp.shape,
                csr=(A_sp.indptr.astype("int32"),
                     A_sp.indices.astype("int32"),
                     A_sp.data.copy()),
            )
            pet.assemble()
            return PETScMatrix(pet)

        from block import block_mat as bmat
        self.M = bmat([[to_dolfin(M_00), to_dolfin(M_01)],
                       [to_dolfin(M_10), to_dolfin(M_11)]])

        # --- variational forms ---
        a = block_form(self.W, 2)
        L = block_form(self.W, 1)

        # 3D: diffusion inside the domain. With exterior='penalty' the DOFs in
        # the 111 region appear in no other term, so they need the penalty to
        # get a nonzero diagonal; with exterior='restrict' they do not exist.
        a[0][0] = (
            k3 * inner(grad(u), grad(v)) * self.d_omega(222)
            + k3 * inner(u, v)           * self.d_omega(222)
        )
        if self.exterior == "penalty":
            a[0][0] += Constant(self.penalty) * inner(u, v) * self.d_omega(111)

        # 1D: diffusion + Nitsche inlet BC
        a[1][1] = k1_f * inner(grad(p), grad(q)) * dx_ + (
            - inner(dot(grad(p), n_fct), q) * ds(tag, domain=self.meshQ)
            - inner(p, dot(grad(q), n_fct)) * ds(tag, domain=self.meshQ)
            + beta * (h_E ** -1) * inner(p, q) * ds(tag, domain=self.meshQ)
        )
        L[0] = inner(Constant(0), v) * self.d_omega(222)
        if self.exterior == "penalty":
            L[0] += Constant(self.penalty) * inner(u_out, v) * self.d_omega(111)
        L[1] = (
            - inner(p_in, dot(grad(q), n_fct)) * ds(tag, domain=self.meshQ)
            + beta * (h_E ** -1) * inner(p_in, q) * ds(tag, domain=self.meshQ)
        )

        self.AD = ii_assemble(a)
        self.b  = ii_assemble(L)

        if self.exterior == "dirichlet":
            # Exact elimination of the exterior DOFs: zero row + zero column, 1
            # on the diagonal, 0 on the rhs. Unlike the penalty, this touches
            # ONLY the DOFs with no support in a 222 cell, so it cannot
            # contaminate the interface. u=0 there contributes nothing to any
            # interior equation, so the interior solution is exactly the one
            # exterior='restrict' computes.
            #
            # M already has zero exterior rows/cols (C's columns were zeroed
            # above), so eliminating in AD is enough for A = AD + M.
            self._eliminate_exterior(self.AD, self.b)
        self.A  = self.AD + self.M
        print("System assembled.")

    def _find_exterior_dofs(self, V):
        """
        Split V's dofs into those with support in at least one 222 cell and
        those with none. Only the latter are eliminated: a dof on the 111/222
        interface DOES have interior support and must be left alone (this is
        exactly what the penalty formulation gets wrong).
        """
        dm      = V.dofmap()
        n       = V.dim()
        has_int = np.zeros(n, dtype=bool)

        for cell in cells(self.meshV):
            if self.V_cell_markers[cell] == 222:
                has_int[dm.cell_dofs(cell.index())] = True

        self.int_dofs = np.flatnonzero(has_int)
        self.ext_dofs = np.flatnonzero(~has_int)
        print(f"Dirichlet elimination: {len(self.ext_dofs)} exterior dofs "
              f"removed | {len(self.int_dofs)} kept "
              f"({len(self.int_dofs)/n*100:.1f}%)")

        if len(self.ext_dofs) == 0:
            raise RuntimeError(
                "exterior='dirichlet' found no exterior dofs — every dof has "
                "interior support, so there is nothing to eliminate."
            )

    def _eliminate_exterior(self, AD, b):
        """
        Zero the rows and columns of AD[0][0] on self.ext_dofs, put 1 on the
        diagonal, and zero b[0] there. Symmetric, so the block stays SPD.
        """
        A00 = as_backend_type(ii_convert(AD[0][0])).mat()
        Asp = csr_matrix(A00.getValuesCSR()[::-1], shape=A00.getSize()).tocoo()

        n       = Asp.shape[0]
        is_ext  = np.zeros(n, dtype=bool)
        is_ext[self.ext_dofs] = True

        # Drop every nonzero whose row OR column is an eliminated dof, then add
        # a unit diagonal there. Done on the COO triplets so we never densify:
        # a fancy-indexed row/col assignment on LIL would materialise an
        # (n_ext x n) dense block.
        keep = ~(is_ext[Asp.row] | is_ext[Asp.col])
        rows = np.concatenate([Asp.row[keep], self.ext_dofs])
        cols = np.concatenate([Asp.col[keep], self.ext_dofs])
        vals = np.concatenate([Asp.data[keep], np.ones(len(self.ext_dofs))])

        Asp = csr_matrix((vals, (rows, cols)), shape=(n, n))
        Asp.eliminate_zeros()

        pet = PETSc.Mat().createAIJ(
            size=Asp.shape,
            csr=(Asp.indptr.astype("int32"),
                 Asp.indices.astype("int32"),
                 Asp.data.copy()),
        )
        pet.assemble()
        AD[0][0] = PETScMatrix(pet)

        # rhs: u = 0 on the eliminated dofs. L[0] only integrates over 222, so
        # these entries should already be ~0 — assert rather than assume.
        ext = self.ext_dofs
        b0  = ii_convert(b[0])
        arr = b0.get_local()
        big = np.abs(arr[ext]).max() if len(ext) else 0.0
        if big > 1e-12:
            raise RuntimeError(
                f"rhs is {big:.3e} on an eliminated exterior dof — expected 0. "
                f"The 3D rhs should integrate over 222 only."
            )
        arr[ext] = 0.0
        b0.set_local(arr)
        b0.apply("insert")
        b[0] = b0

    def _split_solution(self):
        dimV          = self.W[0].dim()
        self.u3d      = Function(self.W[0])
        self.u1d      = Function(self.W[1])
        self.u3d.vector()[:] = self.x_np[:dimV]
        self.u1d.vector()[:] = self.x_np[dimV:]

    # =========================================================================
    # HazNics solver
    # =========================================================================

    @staticmethod
    def _solve_haznics(W, A, b, AD, M, C):
        def block_to_haz(AA):
            if hasattr(AA, "block_collapse"):
                AA = AA.block_collapse()
            brow, bcol = AA.blocks.shape
            for i in range(brow):
                for j in range(bcol):
                    AA[i][j] = ii_collapse(AA[i][j])
            return block_mat_to_block_dCSRmat(AA)

        dimW  = sum(VV.dim() for VV in W)
        bb    = ii_convert(b)
        b_np  = bb[:]
        bhaz  = haznics.create_dvector(b_np)
        xhaz  = haznics.dvec_create_p(dimW)
        Ahaz  = block_to_haz(A)
        Mhaz  = block_to_haz(M)
        ADhaz = block_to_haz(AD)

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
                    f"'{attr}' not available — call build() first."
                )

    def _print_summary(self):
        dimV, dimQ = self.W[0].dim(), self.W[1].dim()
        print("=" * 60)
        print(f"sigma3d={self.sigma3d}  sigma1d_ref={self.sigma1d_ref}  "
              f"kappa={self.kappa}")
        print(f"dim(V)={dimV}  dim(Q)={dimQ}  "
              f"hmax(V)={self.W[0].mesh().hmax():.3f}  "
              f"hmin(Q)={self.W[1].mesh().hmin():.5f}  "
              f"niters={self.niters}  time={self.solve_time:.2f}s")
        print("=" * 60)

    def __repr__(self):
        return (
            f"Solver3D1D(n={self.n}, "
            f"sigma3d={self.sigma3d}, sigma1d_ref={self.sigma1d_ref}, "
            f"kappa={self.kappa}, "
            f"built={self.W is not None})"
        )