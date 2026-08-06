import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from dolfin import (
    Function,
    FunctionSpace,
    TestFunction,
    TrialFunction,
    Constant,
    Measure,
    MeshFunction,
    Point,
    Cell,
    assemble,
    inner,
    grad,
    as_backend_type,
    cells,
    vertex_to_dof_map,
)
from xii import Circle
from xii.linalg.matrix_utils import petsc_serial_matrix


# =============================================================================
# Helpers
# =============================================================================

def dolfin_matrix_to_csr(A_dolfin):
    mat = as_backend_type(A_dolfin).mat()
    indptr, indices, data = mat.getValuesCSR()
    return csr_matrix((data, indices, indptr), shape=mat.getSize())


def average_matrix_diff_radii_skip(V, TV, tv_radii, active_vertices=None):
    """
    Averaging matrix C from 3D V to 1D TV using circle quadrature at each 1D
    vertex. If a vertex has no valid quadrature point inside the 3D mesh,
    that row is skipped (left to zero) instead of raising.

    Returns
    -------
    C_petsc : petsc4py.PETSc.Mat
        Matrix with shape (TV.dim(), V.dim()).
    skipped_vertices : list[int]
        1D vertex ids with no 3D support.
    """
    mesh_x = TV.mesh().coordinates()
    value_size = TV.ufl_element().value_size()
    mesh = V.mesh()
    tree = mesh.bounding_box_tree()
    limit = mesh.num_cells()
    TV_coordinates = TV.tabulate_dof_coordinates().reshape((TV.dim(), -1))
    line_mesh = TV.mesh()
    TV_dm = TV.dofmap()
    V_dm = V.dofmap()

    if value_size > 1:
        TV_dm = TV.sub(0).dofmap()

    Vel = V.element()
    basis_values = np.zeros(V.element().space_dimension() * value_size)

    n_vtx = line_mesh.num_vertices()
    tangent = np.zeros((n_vtx, 3))
    r_sum = np.zeros(n_vtx)
    r_cnt = np.zeros(n_vtx)

    for a, b in line_mesh.cells():
        a, b = int(a), int(b)
        e = mesh_x[a] - mesh_x[b]
        nrm = np.linalg.norm(e)
        if nrm > 0:
            e = e / nrm
        for v in (a, b):
            tangent[v] += e if np.dot(tangent[v], e) >= 0 else -e
        R = max(0.5 * (tv_radii[a] + tv_radii[b]), 0.005)
        for v in (a, b):
            r_sum[v] += R
            r_cnt[v] += 1

    v2d = vertex_to_dof_map(TV)

    if active_vertices is None:
        active_vertices = np.arange(n_vtx, dtype=int)
    else:
        active_vertices = np.asarray(active_vertices, dtype=int).ravel()
        if active_vertices.size == 0:
            active_vertices = np.zeros(0, dtype=int)

    skipped_vertices = []

    with petsc_serial_matrix(TV, V) as mat:
        for v in active_vertices:
            nrm = np.linalg.norm(tangent[v])
            if nrm < 1e-14:
                a, b = line_mesh.cells()[0]
                normal = mesh_x[int(a)] - mesh_x[int(b)]
            else:
                normal = tangent[v] / nrm

            Ri = r_sum[v] / max(r_cnt[v], 1)
            shape = Circle(radius=Ri, degree=10)
            scalar_row = int(v2d[v])
            avg_point = TV_coordinates[scalar_row]

            quadrature = shape.quadrature(avg_point, normal)
            integration_points = quadrature.points
            wq = quadrature.weights

            data = {}
            used_measure = 0.0
            for index, ip in enumerate(integration_points):
                c = tree.compute_first_entity_collision(Point(*ip))
                if c >= limit:
                    continue
                used_measure += wq[index]
                Vcell = Cell(mesh, c)
                vertex_coordinates = Vcell.get_vertex_coordinates()
                cell_orientation = Vcell.orientation()
                basis_values[:] = Vel.evaluate_basis_all(ip, vertex_coordinates, cell_orientation)
                cols_ip = V_dm.cell_dofs(c)
                values_ip = basis_values * wq[index]
                for col, value in zip(cols_ip, values_ip.reshape((-1, value_size))):
                    if col in data:
                        data[col] += value
                    else:
                        data[col] = value.copy()

            if used_measure <= 0.0:
                skipped_vertices.append(v)
                continue

            column_indices = np.array(list(data.keys()), dtype="int32")
            for shift in range(value_size):
                row = scalar_row + shift
                column_values = np.array([data[col][shift] / used_measure for col in column_indices])
                mat.setValues([row], column_indices, column_values)

    return mat, skipped_vertices


# =============================================================================
# Partition solver
# =============================================================================


class SolverPartitionDomain:
    """
    3D-only partition solver with known 1D graph pressure.

    Solves on a partition mesh:

        (AD00 + gamma * C^T G C) u = b0 + gamma * C^T G p_known

    where:
    - AD00 is the 3D operator on the provided partition mesh.
    - C is the 3D->1D averaging operator.
    - G is the 1D coupling mass matrix on the provided graph mesh, weighted per
      cell by the vessel perimeter kappa * 2*pi*R (matching Solver3D1D).
    - p_known is a known graph solution on Q.

    Because kappa is already carried inside G's weighting, `gamma` must be left
    at 1.0 to reproduce the full-domain operator; it is kept only as a knob for
    deliberately rescaling the coupling block.
    """

    def __init__(
        self,
        meshV,
        meshV_markers,
        meshQ,
        q_radii,
        p_known,
        active_q_vertices=None,
        sigma3d=1e-3,
        kappa=1.0,
        gamma=1.0,
        interior_tag=222,
        f3d=0.0,
        C_global=None,
        l2g=None,
    ):
        self.meshV = meshV
        self.meshV_markers = meshV_markers
        self.meshQ = meshQ
        self.q_radii = q_radii
        self.p_known = p_known
        self.active_q_vertices = active_q_vertices
        # Optional: restrict the GLOBAL coupling operator to this box's columns
        # instead of rebuilding it by local circle quadrature. See build().
        self.C_global = C_global
        self.l2g = None if l2g is None else np.asarray(l2g, dtype=int)
        self.sigma3d = float(sigma3d)
        self.kappa = float(kappa)
        self.gamma = float(gamma)
        self.interior_tag = int(interior_tag)
        self.f3d = f3d

        self.V = None
        self.Q = None
        self.d_omega = None

        self.AD00 = None
        self.b0 = None
        self.C = None
        self.G = None
        self.M00 = None
        self.A = None
        self.rhs = None
        self.rhs_b0 = None
        self.rhs_coupling = None

        self.u3d = None
        self.skipped_q_vertices = None
        self.active_q_vertices_count = 0
        self.C_restriction_dropped = None
        self.C_restriction_kept = None
        self.C_rows_straddling = None

    def build(self):
        self.V = FunctionSpace(self.meshV, "CG", 1)
        self.Q = FunctionSpace(self.meshQ, "CG", 1)
        self.d_omega = Measure("dx", domain=self.meshV, subdomain_data=self.meshV_markers)

        u = TrialFunction(self.V)
        v = TestFunction(self.V)

        if isinstance(self.f3d, (int, float)):
            f3d = Constant(float(self.f3d))
        else:
            f3d = self.f3d

        a00_form = self.sigma3d * (inner(grad(u), grad(v)) + inner(u, v)) * self.d_omega(self.interior_tag)
        b0_form = inner(f3d, v) * self.d_omega(self.interior_tag)

        self.AD00 = assemble(a00_form)
        self.b0 = assemble(b0_form)

        q_radii_arr = self._as_q_radii_array(self.q_radii)

        if self.C_global is not None and self.l2g is not None:
            # --- restricted global C -----------------------------------------
            # Local circle quadrature drops quadrature points that fall outside
            # this box and RENORMALIZES by the surviving arc (used_measure in
            # average_matrix_diff_radii_skip). For a vessel whose circle
            # straddles a cut plane that yields a genuinely different average of
            # the same field than the global operator computed -- measured at
            # 77% relative discrepancy -- so A_i is not the one-sided
            # restriction of the global operator and b_i - A_i u* picks up a
            # volume term wherever vessels live. That breaks the Key point
            # eq. (4) relies on.
            #
            # Restricting the global C to this box's columns instead keeps every
            # surviving weight EXACTLY as the global solve computed it. The
            # meshQ is shared across subdomains (decomposeDomain hands every box
            # the same global meshQ), so rows need no mapping at all -- this is
            # a pure column selection.
            Cg = self.C_global.tocsr()
            n_V_local = self.V.dim()
            Ccoo = Cg.tocoo()
            # Global column -> local column, -1 where the box does not own it.
            g2l = -np.ones(Cg.shape[1], dtype=np.int64)
            g2l[self.l2g] = np.arange(self.l2g.size)
            mapped = g2l[Ccoo.col]
            keep = mapped >= 0
            self.C = csr_matrix(
                (Ccoo.data[keep], (Ccoo.row[keep], mapped[keep])),
                shape=(self.Q.dim(), n_V_local),
            )
            self.C.eliminate_zeros()

            # Weight lost to columns outside the box: this is the part of each
            # vessel's circular average that lives in a NEIGHBOUR. It is exactly
            # the 1D-network coupling across the cut, and it is not local data.
            self.C_restriction_dropped = float(np.abs(Ccoo.data[~keep]).sum())
            self.C_restriction_kept = float(np.abs(Ccoo.data[keep]).sum())
            # Rows with any weight outside the box -- vertices whose circle
            # straddles a cut plane.
            rows_out = np.unique(Ccoo.row[~keep])
            rows_in = np.unique(Ccoo.row[keep])
            self.C_rows_straddling = np.intersect1d(rows_out, rows_in)
            self.skipped_q_vertices = []
            self.active_q_vertices_count = int(rows_in.size)
        else:
            C_petsc, skipped = average_matrix_diff_radii_skip(
                self.V,
                self.Q,
                q_radii_arr,
                active_vertices=self.active_q_vertices,
            )
            self.skipped_q_vertices = skipped
            if self.active_q_vertices is None:
                self.active_q_vertices_count = int(self.meshQ.num_vertices())
            else:
                self.active_q_vertices_count = int(np.asarray(self.active_q_vertices).size)

            c_indptr, c_idx, c_data = C_petsc.getValuesCSR()
            self.C = csr_matrix((c_data, c_idx, c_indptr), shape=(self.Q.dim(), self.V.dim()))

        p = TrialFunction(self.Q)
        q = TestFunction(self.Q)
        dx_q = Measure("dx", domain=self.meshQ)

        # G must match the full-domain solver exactly: the coupling mass matrix
        # is weighted per 1D cell by the vessel perimeter kappa * 2*pi*R, not
        # unweighted. An unweighted G gives every subdomain the wrong coupling
        # strength by a large, spatially-varying factor.
        DG0 = FunctionSpace(self.meshQ, "DG", 0)
        gamma_f = Function(DG0)
        q_cells = self.meshQ.cells()
        for i in range(self.meshQ.num_cells()):
            cv0, cv1 = q_cells[i]
            Ri = max(0.5 * (q_radii_arr[int(cv0)] + q_radii_arr[int(cv1)]), 0.005)
            gamma_f.vector()[i] = self.kappa * 2 * np.pi * Ri

        G_dolfin = assemble(gamma_f * inner(p, q) * dx_q)
        self.G = dolfin_matrix_to_csr(G_dolfin)

        self.M00 = self.C.T @ self.G @ self.C

        b0_vec = self.b0.get_local()
        p_known_vec = self._as_q_vector(self.p_known)

        self.A = dolfin_matrix_to_csr(self.AD00) + self.gamma * self.M00
        self.rhs_b0 = b0_vec
        self.rhs_coupling = self.gamma * (self.C.T @ (self.G @ p_known_vec))
        self.rhs = self.rhs_b0 + self.rhs_coupling

        if self.skipped_q_vertices:
            print(
                "Coupling operator C: "
                f"skipped {len(self.skipped_q_vertices)} graph vertices "
                f"(out of {self.active_q_vertices_count} active) "
                "with no 3D support in this partition mesh."
            )
        if getattr(self, "C_restriction_dropped", None) is not None:
            tot = self.C_restriction_kept + self.C_restriction_dropped
            print(
                "Coupling operator C (restricted from global): "
                f"{self.active_q_vertices_count} rows with local support, "
                f"{self.C_rows_straddling.size} straddling a cut, "
                f"weight kept {self.C_restriction_kept / max(tot, 1e-30):.1%}"
            )

        return self

    def solve(self):
        if self.A is None or self.rhs is None or self.V is None:
            raise RuntimeError("build() must be called before solve().")

        u_vec = spsolve(self.A.tocsr(), np.asarray(self.rhs, dtype=float))
        self.u3d = Function(self.V)
        self.u3d.vector()[:] = u_vec
        return self

    def save_paraview(self, output_path):
        if self.u3d is None:
            raise RuntimeError("solve() must be called before save_paraview().")

        from dolfin import XDMFFile

        with XDMFFile(output_path) as f:
            f.parameters["flush_output"] = True
            f.parameters["functions_share_mesh"] = True
            self.u3d.rename("u3d_partition", "3D partition pressure")
            f.write(self.u3d)

    def _as_q_radii_array(self, q_radii):
        if hasattr(q_radii, "array"):
            arr = np.asarray(q_radii.array(), dtype=float)
        else:
            arr = np.asarray(q_radii, dtype=float)

        if arr.ndim != 1 or arr.shape[0] != self.meshQ.num_vertices():
            raise ValueError(
                "q_radii must be a 1D array-like (or MeshFunction) with "
                f"length meshQ.num_vertices()={self.meshQ.num_vertices()}."
            )
        if (arr <= 0).any():
            raise ValueError("q_radii contains non-positive values.")
        return arr

    def _as_q_vector(self, p_known):
        if isinstance(p_known, Function):
            vec = p_known.vector().get_local()
            if vec.shape[0] != self.Q.dim():
                raise ValueError(
                    "p_known Function has incompatible size: "
                    f"{vec.shape[0]} != Q.dim()={self.Q.dim()}"
                )
            return vec.astype(float, copy=False)

        arr = np.asarray(p_known, dtype=float).ravel()
        if arr.shape[0] != self.Q.dim():
            raise ValueError(
                "p_known must be a Function on Q or a vector with length "
                f"Q.dim()={self.Q.dim()}, got {arr.shape[0]}."
            )
        return arr
