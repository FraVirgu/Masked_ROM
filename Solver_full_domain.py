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
    Averaging matrix reducing g in V to TV by integration over the vessel
    cross-section: a circle of radius R(s) centred at each point s of the 1D
    mesh, lying in the plane normal to the centerline.
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

    for idx_c, (a, b) in enumerate(line_mesh.cells()):
        a, b = int(a), int(b)
        e = mesh_x[a] - mesh_x[b]
        nrm = np.linalg.norm(e)
        if nrm > 0:
            e = e / nrm
        for v in (a, b):
            tangent[v] += e if np.dot(tangent[v], e) >= 0 else -e
        R = max(0.5 * (TV_radii[a] + TV_radii[b]), 0.005)
        for v in (a, b):
            r_sum[v] += R
            r_cnt[v] += 1

    v2d = vertex_to_dof_map(TV)

    with petsc_serial_matrix(TV, V) as mat:
        for v in tqdm.tqdm(range(n_vtx), desc=f"Averaging over {n_vtx} vertices", total=n_vtx):
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
                raise RuntimeError(
                    f"1D vertex {v} at {avg_point}: every quadrature point of its averaging circle "
                    f"(R={Ri:.5f}) fell outside the 3D mesh. The centerline must lie inside the domain."
                )

            column_indices = np.array(list(data.keys()), dtype="int32")
            for shift in range(value_size):
                row = scalar_row + shift
                column_values = np.array([data[col][shift] / used_measure for col in column_indices])
                mat.setValues([row], column_indices, column_values, PETSc.InsertMode.INSERT_VALUES)
    return mat


# =============================================================================
# Solver3D1D
# =============================================================================

class Solver3D1D:
    """
    Solves the 3D-1D coupled oxygen perfusion equation on a supplied full
    domain mesh. The full domain can be provided directly as a mesh and its
    cell markers, or it can be inferred from a Boundary object.
    """

    def __init__(
        self,
        path_to_1D_mesh: str,
        full_domain_mesh=None,
        full_domain_markers=None,
        boundary: Boundary | None = None,
        n: int = 10,
        sigma3d: float = 1e-3,
        sigma1d: float = 1.0,
        kappa: float = 1.0,
        beta_nitsche: float = 1.0,
        inlet_tag: int = 111,
        exterior: str = "dirichlet",
    ):
        if exterior != "dirichlet":
            raise ValueError(
                f"Solver3D1D now supports only exterior='dirichlet', got {exterior!r}."
            )

        self.path_to_1D_mesh = path_to_1D_mesh
        self.full_domain_mesh = full_domain_mesh
        self.full_domain_markers = full_domain_markers
        self.boundary = boundary
        self.n = n
        self.sigma3d = sigma3d
        self.sigma1d_ref = sigma1d
        self.kappa = kappa
        self.beta_nitsche = beta_nitsche
        self.inlet_tag = inlet_tag

        self.meshV = None
        self.meshQ = None
        self.Q_markers = None
        self.Q_radii = None
        self.V_cell_markers = None
        self.d_omega = None
        self.ds = None
        self.meshV_full = None
        self.parent_vertex = None
        self.ext_dofs = None
        self.int_dofs = None
        self.W = None
        self.AD = None
        self.M = None
        self.A = None
        self.b = None
        self.C = None
        self.G = None
        self.C_dropped = 0.0
        self.V_DOF = None

        self.x_np = None
        self.u3d = None
        self.u1d = None
        self.niters = None
        self.solve_time = None
        self.pressure_path_report = None

    def build(self):
        self._load_meshes()
        self._assemble_system()
        return self

    def solve(self):
        self._require("A", "b", "W", "C", "AD", "M")

        bb_norm = ii_convert(self.b).norm("l2")
        print(f"‖b‖ = {bb_norm:.6e}")
        if bb_norm == 0.0:
            raise RuntimeError("RHS is zero — inlet marker (tag=111) not found. Check markers.xdmf.")

        t0 = time.time()
        self.niters, _, self.x_np = self._solve_haznics(self.W, self.A, self.b, self.AD, self.M, self.C)
        self.solve_time = time.time() - t0
        self._split_solution()
        self.pressure_path_report = self._print_pressure_path_diagnostics()
        print(f"Solved in {self.solve_time:.3f}s  |  iters: {self.niters}")
        return self

    def save(self, output_folder: str):
        self._require("x_np", "u3d", "u1d")
        os.makedirs(output_folder, exist_ok=True)
        np.save(f"{output_folder}/solution.npy", self.x_np)
        print(f"Solution saved → {output_folder}/solution.npy")
        self._print_summary()

    def save_paraview(self, output_folder: str):
        self._require("u3d", "u1d", "meshV", "meshQ")
        os.makedirs(output_folder, exist_ok=True)

        with XDMFFile(f"{output_folder}/u3d.xdmf") as f:
            f.parameters["flush_output"] = True
            f.parameters["functions_share_mesh"] = True
            self.u3d.rename("u3d", "3D pressure")
            f.write(self.u3d)

        with XDMFFile(f"{output_folder}/u1d.xdmf") as f:
            f.parameters["flush_output"] = True
            f.parameters["functions_share_mesh"] = True
            self.u1d.rename("u1d", "1D vessel pressure")
            f.write(self.u1d)

        with XDMFFile(f"{output_folder}/cell_markers.xdmf") as f:
            f.parameters["flush_output"] = True
            f.write(self.V_cell_markers)

        with XDMFFile(f"{output_folder}/vessel_markers.xdmf") as f:
            f.parameters["flush_output"] = True
            f.write(self.Q_markers)

        Q = self.W[1]
        edge_radii = MeshFunction("double", self.meshQ, 1, 0.0)
        for cell in cells(self.meshQ):
            v0, v1 = cell.entities(0)
            edge_radii[cell.index()] = 0.5 * (self.Q_radii[int(v0)] + self.Q_radii[int(v1)])
        with XDMFFile(f"{output_folder}/vessel_radii.xdmf") as f:
            f.parameters["flush_output"] = True
            edge_radii.rename("radius", "vessel radius")
            f.write(edge_radii)

        radius_fn = Function(Q)
        for v_idx in range(self.meshQ.num_vertices()):
            radius_fn.vector()[v_idx] = self.Q_radii[v_idx]
        with XDMFFile(f"{output_folder}/vessel_radii_smooth.xdmf") as f:
            f.parameters["flush_output"] = True
            f.parameters["functions_share_mesh"] = True
            radius_fn.rename("radius_smooth", "vessel radius (smooth)")
            f.write(radius_fn)

        if self.pressure_path_report:
            report_path = f"{output_folder}/pressure_path_diagnostics.txt"
            with open(report_path, "w") as report_file:
                report_file.write(self.pressure_path_report)

        self._print_summary()
        print(f"ParaView files → {output_folder}/")
        for fname in [
            "u3d.xdmf",
            "u1d.xdmf",
            "cell_markers.xdmf",
            "vessel_markers.xdmf",
            "vessel_radii.xdmf",
            "vessel_radii_smooth.xdmf",
        ]:
            print(f"  {fname}")

    def _load_meshes(self):
        if self.full_domain_mesh is None:
            if self.boundary is None:
                self.meshV = BoxMesh(Point(-1, -1, -1), Point(1, 1, 1), self.n, self.n, self.n)
            else:
                bbox = self.boundary._bbox
                (xmin, xmax), (ymin, ymax), (zmin, zmax) = bbox
                self.meshV = BoxMesh(Point(xmin, ymin, zmin), Point(xmax, ymax, zmax), self.n, self.n, self.n)
        else:
            self.meshV = self.full_domain_mesh

        self.meshV_full = self.meshV

        if self.full_domain_markers is not None:
            self.V_cell_markers = self.full_domain_markers
        else:
            self.V_cell_markers = MeshFunction("size_t", self.meshV, 3, 111)
            inside_count = 0
            for cell in cells(self.meshV):
                mp = cell.midpoint()
                if self.boundary is None or self.boundary([mp.x(), mp.y(), mp.z()]):
                    self.V_cell_markers[cell] = 222
                    inside_count += 1

            total = self.meshV.num_cells()
            print(f"3D mesh: {total} cells | inside domain: {inside_count} ({inside_count / total * 100:.1f}%)")

        self.d_omega = Measure("dx", domain=self.meshV, subdomain_data=self.V_cell_markers)

        self.meshQ = Mesh()
        with XDMFFile(f"{self.path_to_1D_mesh}marked_mesh.xdmf") as f:
            f.read(self.meshQ)

        self.Q_markers = MeshFunction("size_t", self.meshQ, 0)
        xdmf_m = XDMFFile(f"{self.path_to_1D_mesh}markers.xdmf")
        xdmf_m.read(self.Q_markers)
        xdmf_m.close()

        radii_path = f"{self.path_to_1D_mesh}radii.xdmf"
        if not os.path.exists(radii_path):
            raise RuntimeError(f"radii.xdmf not found at '{radii_path}'. Run CCOVascularMesh.export_xdmf() first.")
        self.Q_radii = MeshFunction("double", self.meshQ, 0)
        xdmf_r = XDMFFile(radii_path)
        xdmf_r.read(self.Q_radii)
        xdmf_r.close()
        r_arr = self.Q_radii.array()
        if (r_arr <= 0).any():
            raise RuntimeError(f"radii.xdmf contains {(r_arr <= 0).sum()} non-positive radii.")
        print(f"Q_radii: min={r_arr.min():.6f}  max={r_arr.max():.6f}")

        self.ds = Measure("ds", domain=self.meshQ, subdomain_data=self.Q_markers)

        print(f"1D mesh: {self.meshQ.num_cells()} edges | {self.meshQ.num_vertices()} vertices")
        self._print_node_radius_ranges()

    def _print_node_radius_ranges(self):
        """Print the global node-radius interval over the 1D graph."""
        n_v = self.meshQ.num_vertices()
        if n_v == 0:
            print("Node radii: empty 1D mesh")
            return

        node_min = np.full(n_v, np.inf, dtype=float)
        node_max = np.full(n_v, -np.inf, dtype=float)
        r_vertex = np.asarray(self.Q_radii.array(), dtype=float)

        for a, b in self.meshQ.cells():
            a = int(a)
            b = int(b)
            r_edge = 0.5 * (r_vertex[a] + r_vertex[b])
            if r_edge < node_min[a]:
                node_min[a] = r_edge
            if r_edge > node_max[a]:
                node_max[a] = r_edge
            if r_edge < node_min[b]:
                node_min[b] = r_edge
            if r_edge > node_max[b]:
                node_max[b] = r_edge

        # Fallback for isolated vertices: interval collapses to nodal value.
        isolated = ~np.isfinite(node_min)
        node_min[isolated] = r_vertex[isolated]
        node_max[isolated] = r_vertex[isolated]

        r_min = float(node_min.min())
        r_max = float(node_max.max())
        print(f"Node radii (1D graph): r belongs [{r_min:.6f}, {r_max:.6f}]")

    def _assemble_system(self):
        self._require("meshV", "meshQ", "ds", "Q_radii")

        V = FunctionSpace(self.meshV, "CG", 1)
        Q = FunctionSpace(self.meshQ, "CG", 1)
        self.W = [V, Q]
        self.V_DOF = V.dofmap().global_dimension()
        print(f"3D DOFs: {self.V_DOF} | 1D DOFs: {Q.dofmap().global_dimension()}")

        self._find_exterior_dofs(V)

        u, v = TrialFunction(V), TestFunction(V)
        p, q = TrialFunction(Q), TestFunction(Q)
        ds = self.ds
        tag = self.inlet_tag
        k3 = Constant(self.sigma3d)
        beta = Constant(self.beta_nitsche)
        h_E = MaxCellEdgeLength(self.meshQ)
        n_fct = FacetNormal(self.meshQ)
        p_in = Constant(1.0)
        dx_ = Measure("dx", domain=self.meshQ)

        n_V = V.dofmap().global_dimension()
        n_Q = Q.dofmap().global_dimension()

        C_petsc = average_matrix_diff_radii(V, Q, self.Q_radii)
        indptr, idx, data_ = C_petsc.getValuesCSR()
        C = csr_matrix((data_, idx, indptr), shape=(n_Q, n_V))

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
        self.C_dropped = float(lost)
        print(f"C: zeroed {len(self.ext_dofs)} exterior columns (dropped weight {lost:.3e})")

        self.C = C

        DG0 = FunctionSpace(self.meshQ, "DG", 0)
        gamma_f = Function(DG0)
        k1_f = Function(DG0)
        for i in range(self.meshQ.num_cells()):
            cv0, cv1 = self.meshQ.cells()[i]
            Ri = max(0.5 * (self.Q_radii[int(cv0)] + self.Q_radii[int(cv1)]), 0.005)
            gamma_f.vector()[i] = self.kappa * 2 * np.pi * Ri
            k1_f.vector()[i] = self.sigma1d_ref * np.pi * Ri ** 2

        G_dolfin = assemble(gamma_f * inner(p, q) * dx_)
        gi, gj, gv = as_backend_type(G_dolfin).mat().getValuesCSR()
        G = csr_matrix((gv, gj, gi), shape=(n_Q, n_Q))
        self.G = G

        M_00 = C.T @ G @ C
        M_01 = -C.T @ G
        M_10 = -G @ C
        M_11 = G

        def to_dolfin(A_sp):
            A_sp = csr_matrix(A_sp)
            pet = PETSc.Mat().createAIJ(
                size=A_sp.shape,
                csr=(A_sp.indptr.astype("int32"), A_sp.indices.astype("int32"), A_sp.data.copy()),
            )
            pet.assemble()
            return PETScMatrix(pet)

        from block import block_mat as bmat
        self.M = bmat([[to_dolfin(M_00), to_dolfin(M_01)], [to_dolfin(M_10), to_dolfin(M_11)]])

        a = block_form(self.W, 2)
        L = block_form(self.W, 1)

        a[0][0] = (
            k3 * inner(grad(u), grad(v)) * self.d_omega(222)
            + k3 * inner(u, v) * self.d_omega(222)
        )

        a[1][1] = k1_f * inner(grad(p), grad(q)) * dx_ + (
            -inner(dot(grad(p), n_fct), q) * ds(tag, domain=self.meshQ)
            -inner(p, dot(grad(q), n_fct)) * ds(tag, domain=self.meshQ)
            + beta * (h_E ** -1) * inner(p, q) * ds(tag, domain=self.meshQ)
        )
        L[0] = inner(Constant(0), v) * self.d_omega(222)
        L[1] = (
            -inner(p_in, dot(grad(q), n_fct)) * ds(tag, domain=self.meshQ)
            + beta * (h_E ** -1) * inner(p_in, q) * ds(tag, domain=self.meshQ)
        )

        self.AD = ii_assemble(a)
        self.b = ii_assemble(L)

        self._eliminate_exterior(self.AD, self.b)
        self.A = self.AD + self.M
        print("System assembled.")

    def _find_exterior_dofs(self, V):
        dm = V.dofmap()
        n = V.dim()
        has_int = np.zeros(n, dtype=bool)

        for cell in cells(self.meshV):
            if self.V_cell_markers[cell] == 222:
                has_int[dm.cell_dofs(cell.index())] = True

        self.int_dofs = np.flatnonzero(has_int)
        self.ext_dofs = np.flatnonzero(~has_int)
        print(f"Dirichlet elimination: {len(self.ext_dofs)} exterior dofs removed | {len(self.int_dofs)} kept ({len(self.int_dofs) / n * 100:.1f}%)")

        if len(self.ext_dofs) == 0:
            print("WARNING: exterior='dirichlet' found no exterior dofs — every dof has interior support. Skipping elimination.")

    def _eliminate_exterior(self, AD, b):
        A00 = as_backend_type(ii_convert(AD[0][0])).mat()
        Asp = csr_matrix(A00.getValuesCSR()[::-1], shape=A00.getSize()).tocoo()

        n = Asp.shape[0]
        is_ext = np.zeros(n, dtype=bool)
        is_ext[self.ext_dofs] = True

        keep = ~(is_ext[Asp.row] | is_ext[Asp.col])
        rows = np.concatenate([Asp.row[keep], self.ext_dofs])
        cols = np.concatenate([Asp.col[keep], self.ext_dofs])
        vals = np.concatenate([Asp.data[keep], np.ones(len(self.ext_dofs))])

        Asp = csr_matrix((vals, (rows, cols)), shape=(n, n))
        Asp.eliminate_zeros()

        pet = PETSc.Mat().createAIJ(size=Asp.shape, csr=(Asp.indptr.astype("int32"), Asp.indices.astype("int32"), Asp.data.copy()))
        pet.assemble()
        AD[0][0] = PETScMatrix(pet)

        ext = self.ext_dofs
        b0 = ii_convert(b[0])
        arr = b0.get_local()
        big = np.abs(arr[ext]).max() if len(ext) else 0.0
        if big > 1e-12:
            raise RuntimeError(f"rhs is {big:.3e} on an eliminated exterior dof — expected 0. The 3D rhs should integrate over 222 only.")
        arr[ext] = 0.0
        b0.set_local(arr)
        b0.apply("insert")
        b[0] = b0

    def _split_solution(self):
        dimV = self.W[0].dim()
        self.u3d = Function(self.W[0])
        self.u1d = Function(self.W[1])
        self.u3d.vector()[:] = self.x_np[:dimV]
        self.u1d.vector()[:] = self.x_np[dimV:]

    def _print_pressure_path_diagnostics(self):
        """Print pressure values and drops along inlet-to-outlet tree paths."""
        if self.meshQ is None or self.Q_markers is None or self.u1d is None:
            return None

        import networkx as nx

        mesh = self.meshQ
        v2d = vertex_to_dof_map(self.W[1])
        values = self.u1d.vector().get_local()[v2d]

        inlet_vertices = [v for v in range(mesh.num_vertices()) if self.Q_markers[v] == self.inlet_tag]
        outlet_vertices = [v for v in range(mesh.num_vertices()) if self.Q_markers[v] == 999]

        lines = []

        def emit(line):
            lines.append(line)

        if len(inlet_vertices) != 1:
            emit(
                f"Pressure path diagnostic skipped: expected 1 inlet vertex, found {len(inlet_vertices)}."
            )
            return "\n".join(lines) + "\n"

        if not outlet_vertices:
            emit("Pressure path diagnostic skipped: no outlet vertices found.")
            return "\n".join(lines) + "\n"

        inlet = inlet_vertices[0]
        G = nx.Graph()
        for a, b in mesh.cells():
            a = int(a)
            b = int(b)
            G.add_edge(a, b, length=float(np.linalg.norm(mesh.coordinates()[a] - mesh.coordinates()[b])))

        emit("")
        emit("=" * 70)
        emit("Pressure drop along inlet-to-outlet paths")
        emit("=" * 70)
        for outlet in outlet_vertices:
            try:
                path = nx.shortest_path(G, source=inlet, target=outlet, weight="length")
            except nx.NetworkXNoPath:
                emit(f"Outlet {outlet}: no path from inlet")
                continue

            emit(f"Outlet {outlet}: {len(path) - 1} edges")
            for a, b in zip(path[:-1], path[1:]):
                pa = float(values[a])
                pb = float(values[b])
                dp = pb - pa
                seg_len = float(np.linalg.norm(mesh.coordinates()[a] - mesh.coordinates()[b]))
                emit(
                    f"  {a:4d} -> {b:4d}  length={seg_len:.4f}  "
                    f"p={pa:.6e} -> {pb:.6e}  dp={dp:+.6e}"
                )
            emit(
                f"  path drop: {float(values[path[-1]] - values[path[0]]):+.6e}"
            )

        return "\n".join(lines) + "\n"


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

        dimW = sum(VV.dim() for VV in W)
        bb = ii_convert(b)
        b_np = bb[:]
        bhaz = haznics.create_dvector(b_np)
        xhaz = haznics.dvec_create_p(dimW)
        Ahaz = block_to_haz(A)
        Mhaz = block_to_haz(M)
        ADhaz = block_to_haz(AD)

        csr0, csr1, csr2 = C.indptr, C.indices, C.data
        Chaz = haznics.create_matrix(csr2, csr1, csr0, C.shape[1])
        niters = haznics.fenics_metric_amg_solver(Ahaz, bhaz, xhaz, ADhaz, Mhaz, Chaz)

        haznics.dvec_write("/tmp/solution_raw.dat", xhaz)
        x_np = np.loadtxt("/tmp/solution_raw.dat", skiprows=1)
        return niters, xhaz, x_np

    def _require(self, *attrs):
        for attr in attrs:
            if getattr(self, attr) is None:
                raise RuntimeError(f"'{attr}' not available — call build() first.")

    def _print_summary(self):
        dimV, dimQ = self.W[0].dim(), self.W[1].dim()
        print("=" * 60)
        print(f"sigma3d={self.sigma3d}  sigma1d_ref={self.sigma1d_ref}  kappa={self.kappa}")
        print(
            f"dim(V)={dimV}  dim(Q)={dimQ}  hmax(V)={self.W[0].mesh().hmax():.3f}  "
            f"hmin(Q)={self.W[1].mesh().hmin():.5f}  niters={self.niters}  time={self.solve_time:.2f}s"
        )
        print("=" * 60)

    def __repr__(self):
        return f"Solver3D1D(n={self.n}, sigma3d={self.sigma3d}, sigma1d_ref={self.sigma1d_ref}, kappa={self.kappa}, built={self.W is not None})"
