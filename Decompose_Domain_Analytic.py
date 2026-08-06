# =============================================================================
# Entry point
# =============================================================================

import math
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import issparse

from dolfin import (
    BoxMesh,
    Point,
    FunctionSpace,
    Function,
    MeshFunction,
    Measure,
    cells,
    facets,
    TrialFunction,
    TestFunction,
    inner,
    assemble,
    as_backend_type,
)

from Solver_full_domain import Solver3D1D
from Solver_partition_domain import SolverPartitionDomain
from Analytic_Domain import Domain
from Boundary import SphereBoundary, random_sphere_points


def check_sphere_domain_consistency(boundary, n_min, n_max, atol=1e-12):
    if not hasattr(boundary, "_bbox"):
        raise RuntimeError("Boundary has no _bbox; cannot verify consistency.")

    bbox = boundary._bbox
    domain_center = 0.5 * (n_min + n_max)
    domain_len = n_max - n_min

    for axis, (bmin, bmax) in zip(("x", "y", "z"), bbox):
        axis_center = 0.5 * (bmin + bmax)
        axis_len = bmax - bmin

        if not np.isclose(axis_center, domain_center, atol=atol, rtol=0.0):
            raise RuntimeError(
                f"Inconsistent {axis}-center: boundary bbox center={axis_center}, "
                f"domain center={domain_center}."
            )
        if not np.isclose(axis_len, domain_len, atol=atol, rtol=0.0):
            raise RuntimeError(
                f"Inconsistent {axis}-length: boundary bbox length={axis_len}, "
                f"domain length={domain_len}."
            )

    if hasattr(boundary, "radius"):
        expected_len = 2.0 * float(boundary.radius)
        if not np.isclose(domain_len, expected_len, atol=atol, rtol=0.0):
            raise RuntimeError(
                f"Inconsistent sphere radius: 2*radius={expected_len} but "
                f"domain length={domain_len}."
            )


def matrix_to_csr(A_any):
    if isinstance(A_any, csr_matrix):
        return A_any
    if issparse(A_any):
        return A_any.tocsr()

    # petsc4py.Mat-like object
    if hasattr(A_any, "getValuesCSR") and hasattr(A_any, "getSize"):
        indptr, indices, data = A_any.getValuesCSR()
        return csr_matrix((data, indices, indptr), shape=A_any.getSize())

    # dolfin PETScMatrix-like object
    if hasattr(A_any, "mat"):
        mat = A_any.mat()
        indptr, indices, data = mat.getValuesCSR()
        return csr_matrix((data, indices, indptr), shape=mat.getSize())

    if isinstance(A_any, np.ndarray):
        # xii/dolfin block access can return object-wrapped containers.
        # Unwrap object containers recursively before numeric conversion.
        if A_any.dtype == object:
            if A_any.shape == ():
                return matrix_to_csr(A_any.item())
            if A_any.size == 1:
                return matrix_to_csr(A_any.ravel()[0])
            # If this is an object block array, try to extract the unique
            # matrix-like entry (common in xii block wrappers).
            flat = [x for x in A_any.ravel() if x is not None]
            matrix_like = [
                x for x in flat
                if hasattr(x, "mat")
                or (hasattr(x, "getValuesCSR") and hasattr(x, "getSize"))
                or issparse(x)
                or isinstance(x, csr_matrix)
            ]
            if len(matrix_like) == 1:
                return matrix_to_csr(matrix_like[0])
            raise TypeError(
                "Object ndarray does not contain a unique matrix entry; "
                f"shape={A_any.shape}, non_none={len(flat)}, matrix_like={len(matrix_like)}"
            )
        return csr_matrix(np.asarray(A_any, dtype=float))

    if isinstance(A_any, (list, tuple)):
        return matrix_to_csr(np.asarray(A_any))

    try:
        backend = as_backend_type(A_any)
    except Exception:
        backend = None
    if backend is not None and hasattr(backend, "mat"):
        mat = backend.mat()
        indptr, indices, data = mat.getValuesCSR()
        return csr_matrix((data, indices, indptr), shape=mat.getSize())

    raise TypeError(f"Unsupported matrix type for CSR conversion: {type(A_any)!r}")


ARTIFICIAL_FACET_TAG = 77


def mark_artificial_facets(partition_solver, g_min, g_max, tol=1e-9):
    """
    Tag the facets of one subdomain box that lie on an ARTIFICIAL cut plane,
    i.e. on a face of the local box that is not a face of the global mesh.

    Facets on the true outer boundary are left untagged so they keep whatever
    natural condition the local form implies, matching the full-domain solve.

    Returns (facet_markers, n_tagged).
    """
    mesh = partition_solver.meshV
    markers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)

    coords = mesh.coordinates()
    sub_lo = coords.min(axis=0)
    sub_hi = coords.max(axis=0)

    # Which of the 6 box faces are artificial cuts.
    lo_is_cut = [abs(sub_lo[a] - g_min[a]) > tol for a in range(3)]
    hi_is_cut = [abs(sub_hi[a] - g_max[a]) > tol for a in range(3)]

    n_tagged = 0
    for f in facets(mesh):
        if not f.exterior():
            continue
        mp = f.midpoint()
        p = (mp.x(), mp.y(), mp.z())
        for a in range(3):
            if lo_is_cut[a] and abs(p[a] - sub_lo[a]) < tol:
                markers[f] = ARTIFICIAL_FACET_TAG
                n_tagged += 1
                break
            if hi_is_cut[a] and abs(p[a] - sub_hi[a]) < tol:
                markers[f] = ARTIFICIAL_FACET_TAG
                n_tagged += 1
                break

    return markers, n_tagged


def assemble_robin_interface(partition_solver, facet_markers):
    """
    Assemble G_i^Gamma, the artificial-interface mass matrix of eqs. (4)-(5):

        (G_i^Gamma)_kl = \\int_{Gamma_i^art} phi_k phi_l ds

    on the facets tagged by mark_artificial_facets.

    This is a SURFACE integral on the cut planes -- not a volume mass matrix,
    despite the shape. Rows and columns of dofs that touch no artificial facet
    are identically zero, so G_i^Gamma is supported entirely on the interface.

    Because it enters the bilinear form (and its product with a trace enters the
    linear form), both sides stay dimensionally consistent: the correction is a
    state-type quantity, never a load added onto a solution.
    """
    V = partition_solver.V
    u = TrialFunction(V)
    v = TestFunction(V)
    ds_art = Measure(
        "ds", domain=partition_solver.meshV, subdomain_data=facet_markers
    )
    G_dolfin = assemble(inner(u, v) * ds_art(ARTIFICIAL_FACET_TAG))
    return matrix_to_csr(G_dolfin)


def eliminate_exterior_local(
    partition_solver, interior_tag=222, global_ext_mask=None, l2g=None
):
    """
    Apply the same exterior-dof elimination that Solver3D1D does, to one
    subdomain system, in place.

    A dof is interior if it belongs to at least one cell marked `interior_tag`.
    Every other dof lies fully outside the physical domain: its row/column is
    dropped, replaced by a unit diagonal, and its rhs zeroed, pinning u=0 there
    exactly as the full-domain solve does.

    If `global_ext_mask` (a boolean array over global dofs) and `l2g` are given,
    the classification is INHERITED from the global solve instead of being
    recomputed locally. This matters: the local BoxMesh tetrahedralizes
    boundary-straddling cells differently from the global mesh, so the local
    midpoint test disagrees with the global one near the sphere surface. Dofs
    the box eliminates but the global solve keeps carry a substantial u* (up to
    0.35 measured), and since elimination drops their columns, A_i u* silently
    ignores that data and fabricates a residual -- which breaks eq. (4)'s
    extraction with no warning.

    Returns (n_ext, n_int).
    """
    V = partition_solver.V
    markers = partition_solver.meshV_markers
    mesh = partition_solver.meshV
    dm = V.dofmap()
    n = V.dim()

    # Local support: a dof only has a row in A_i if some cell touching it is
    # marked interior, because a00_form integrates over d_omega(interior_tag)
    # alone. A dof with no locally-interior cell has an all-zero row.
    has_int = np.zeros(n, dtype=bool)
    for cell in cells(mesh):
        if markers[cell] == interior_tag:
            has_int[dm.cell_dofs(cell.index())] = True

    is_ext_local = ~has_int

    if global_ext_mask is not None and l2g is not None:
        # Eliminate the UNION: globally-exterior dofs (so A_i drops exactly what
        # the global operator drops, keeping u* zero there) PLUS dofs with no
        # local interior support (whose rows are identically zero -- keeping
        # them makes A_i exactly singular).
        is_ext_global = np.asarray(global_ext_mask, dtype=bool)[
            np.asarray(l2g, dtype=int)
        ]
        # Dofs the global solve keeps but this box cannot support. These are
        # unavoidable: u* is nonzero there, yet A_i has no equation for them,
        # so eq. (4)'s residual will pick up a term the transfer cannot carry.
        partition_solver.unsupported_interior = np.flatnonzero(
            is_ext_local & ~is_ext_global
        )
        is_ext_local = is_ext_local | is_ext_global
    else:
        partition_solver.unsupported_interior = np.zeros(0, dtype=int)

    ext_dofs = np.flatnonzero(is_ext_local)
    int_dofs = np.flatnonzero(~is_ext_local)

    if ext_dofs.size == 0:
        partition_solver.ext_dofs = ext_dofs
        partition_solver.int_dofs = int_dofs
        return 0, int(int_dofs.size)

    Asp = partition_solver.A.tocoo()
    is_ext = np.zeros(n, dtype=bool)
    is_ext[ext_dofs] = True

    keep = ~(is_ext[Asp.row] | is_ext[Asp.col])
    rows = np.concatenate([Asp.row[keep], ext_dofs])
    cols = np.concatenate([Asp.col[keep], ext_dofs])
    vals = np.concatenate([Asp.data[keep], np.ones(ext_dofs.size)])

    A_new = csr_matrix((vals, (rows, cols)), shape=(n, n))
    A_new.eliminate_zeros()

    rhs_new = np.array(partition_solver.rhs, dtype=float)
    rhs_new[ext_dofs] = 0.0

    partition_solver.A = A_new
    partition_solver.rhs = rhs_new
    partition_solver.ext_dofs = ext_dofs
    partition_solver.int_dofs = int_dofs

    return int(ext_dofs.size), int(int_dofs.size)


def solve_partition_domain(
    subdomains, solver, print_summary=True, restrict_global_C=False
):
    """Solve each subdomain in isolation and attach diagnostics.

    This is the "raw" baseline: every box is solved with the natural (Neumann)
    condition on its artificial cut planes, i.e. with no coupling to its
    neighbours. It is the reference any transmission scheme is measured
    against, and the starting point they improve on.

    It must run before any of them: it populates "partition_solver" and the
    per-subdomain error fields they read.

    restrict_global_C=True builds each box's coupling operator by restricting
    the GLOBAL C to that box's columns, instead of re-running circle quadrature
    on the local mesh. The local quadrature clips circles at the box boundary
    and renormalizes by the surviving arc, which makes C_i disagree with the
    global operator by ~77% and destroys the one-sidedness eq. (4) assumes.
    """
    V_global = solver.W[0]
    n_global = V_global.dim()
    u_partition_sum_raw = np.zeros(n_global, dtype=float)
    u_partition_count_raw = np.zeros(n_global, dtype=float)

    # Inherit the exterior classification from the global solve so that each
    # A_i eliminates exactly the dofs the global operator eliminates. Deriving
    # it locally disagrees near the sphere surface (different tetrahedra), which
    # leaves u* nonzero on locally-eliminated dofs.
    global_ext_mask = None
    g_ext = getattr(solver, "ext_dofs", None)
    if g_ext is not None and np.size(g_ext):
        global_ext_mask = np.zeros(n_global, dtype=bool)
        global_ext_mask[np.asarray(g_ext, dtype=int)] = True

    for subdomain in subdomains:
        local_to_global_dof = subdomain["local_to_global_dof"]

        # Restrict graph coupling candidates to vertices near this subdomain.
        q_coords = subdomain["meshQ"].coordinates()
        q_radii_arr = np.asarray(solver.Q_radii.array(), dtype=float)
        rmax = float(q_radii_arr.max()) if q_radii_arr.size else 0.0
        sub_coords = subdomain["meshV"].coordinates()
        xyz_min = sub_coords.min(axis=0) - rmax
        xyz_max = sub_coords.max(axis=0) + rmax
        active_q_vertices = np.where(
            (q_coords[:, 0] >= xyz_min[0]) & (q_coords[:, 0] <= xyz_max[0])
            & (q_coords[:, 1] >= xyz_min[1]) & (q_coords[:, 1] <= xyz_max[1])
            & (q_coords[:, 2] >= xyz_min[2]) & (q_coords[:, 2] <= xyz_max[2])
        )[0].astype(int)
        subdomain["active_q_vertices"] = active_q_vertices

        # Solve the partition 3D problem using known graph pressure on meshQ.
        partition_solver = SolverPartitionDomain(
            meshV=subdomain["meshV"],
            meshV_markers=subdomain["V_cell_markers"],
            meshQ=subdomain["meshQ"],
            q_radii=solver.Q_radii,
            p_known=subdomain["sol_1d"],
            active_q_vertices=active_q_vertices,
            sigma3d=solver.sigma3d,
            kappa=solver.kappa,
            # gamma stays 1.0: kappa is inside G's per-cell weighting, exactly as
            # in Solver3D1D (M_00 = C.T @ G @ C, no extra factor). rho_robin is a
            # Robin penalty and must NOT be used to scale the coupling block.
            gamma=1.0,
            interior_tag=222,
            f3d=0.0,
            C_global=solver.C if restrict_global_C else None,
            l2g=local_to_global_dof if restrict_global_C else None,
        ).build()

        # Pin u=0 outside the physical domain, as the full-domain solve does,
        # using the GLOBAL classification so the two agree dof-for-dof.
        n_ext, n_int = eliminate_exterior_local(
            partition_solver,
            interior_tag=222,
            global_ext_mask=global_ext_mask,
            l2g=local_to_global_dof,
        )
        subdomain["exterior_dofs_eliminated"] = n_ext
        subdomain["interior_dofs_kept"] = n_int

        partition_solver.solve()
        subdomain["partition_solver"] = partition_solver

        u_part_raw = partition_solver.u3d.vector().get_local()
        subdomain["u3d_partition_raw"] = partition_solver.u3d
        subdomain["u3d_partition"] = partition_solver.u3d

        # Local reference-vs-partition error on the subdomain mesh.
        u_full_local = subdomain["sol_3d_sub"].vector().get_local()
        local_error_raw = u_full_local - u_part_raw
        u_full_local_norm = float(np.linalg.norm(u_full_local))
        subdomain["u3d_partition_error_local_raw"] = local_error_raw
        subdomain["u3d_partition_error_local_raw_norm_l2"] = float(np.linalg.norm(local_error_raw))
        subdomain["u3d_partition_error_local_raw_rel_l2"] = (
            subdomain["u3d_partition_error_local_raw_norm_l2"] / u_full_local_norm
            if u_full_local_norm > 1e-12
            else 0.0
        )
        subdomain["u3d_full_local_norm_l2"] = u_full_local_norm
        subdomain["u3d_partition_local_norm_l2"] = float(np.linalg.norm(u_part_raw))
        subdomain["partition_rhs_norm_l2"] = float(np.linalg.norm(partition_solver.rhs))
        subdomain["partition_rhs_b0_norm_l2"] = float(np.linalg.norm(partition_solver.rhs_b0))
        subdomain["partition_rhs_coupling_norm_l2"] = float(
            np.linalg.norm(partition_solver.rhs_coupling)
        )
        subdomain["partition_skipped_q_vertices_count"] = int(
            len(partition_solver.skipped_q_vertices)
            if partition_solver.skipped_q_vertices is not None
            else 0
        )
        subdomain["partition_active_q_vertices_count"] = int(
            partition_solver.active_q_vertices_count
        )

        # Contribute raw partition solution to a reconstructed full-domain field.
        u_partition_sum_raw[local_to_global_dof] += u_part_raw
        u_partition_count_raw[local_to_global_dof] += 1.0

    covered_raw = u_partition_count_raw > 0.0
    u3d_reconstructed_raw_vec = np.zeros(n_global, dtype=float)
    u3d_reconstructed_raw_vec[covered_raw] = (
        u_partition_sum_raw[covered_raw] / u_partition_count_raw[covered_raw]
    )

    covered = covered_raw
    u3d_reconstructed_vec = u3d_reconstructed_raw_vec

    # If a global dof is uncovered (should not happen with full tiling), keep
    # the full-domain reference value to avoid introducing artificial error.
    u3d_full_vec = solver.u3d.vector().get_local()
    u3d_reconstructed_raw_vec[~covered_raw] = u3d_full_vec[~covered_raw]
    u3d_reconstructed_vec[~covered] = u3d_full_vec[~covered]

    u3d_reconstructed_raw = Function(V_global)
    u3d_reconstructed_raw.vector()[:] = u3d_reconstructed_raw_vec

    u3d_reconstructed = Function(V_global)
    u3d_reconstructed.vector()[:] = u3d_reconstructed_vec

    full_ref_l2 = float(np.linalg.norm(u3d_full_vec))

    full_error_raw = u3d_full_vec - u3d_reconstructed_raw_vec
    full_error_raw_l2 = float(np.linalg.norm(full_error_raw))
    full_error_raw_rel_l2 = full_error_raw_l2 / full_ref_l2 if full_ref_l2 > 0.0 else 0.0

    full_error = u3d_full_vec - u3d_reconstructed_vec
    full_error_l2 = float(np.linalg.norm(full_error))
    full_error_rel_l2 = full_error_l2 / full_ref_l2 if full_ref_l2 > 0.0 else 0.0
    covered_count = int(np.count_nonzero(covered))
    uncovered_count = int(n_global - covered_count)
    overlap_count = int(np.count_nonzero(u_partition_count_raw > 1.0))
    max_overlap = (
        int(u_partition_count_raw.max()) if len(u_partition_count_raw) else 0
    )

    for subdomain in subdomains:
        subdomain["u3d_reconstructed_full_raw"] = u3d_reconstructed_raw
        subdomain["u3d_partition_full_error_raw_l2"] = full_error_raw_l2
        subdomain["u3d_partition_full_error_raw_rel_l2"] = full_error_raw_rel_l2
        subdomain["u3d_reconstructed_full"] = u3d_reconstructed
        subdomain["u3d_partition_full_error_l2"] = full_error_l2
        subdomain["u3d_partition_full_error_rel_l2"] = full_error_rel_l2
        subdomain["u3d_partition_global_coverage_count"] = covered_count
        subdomain["u3d_partition_global_uncovered_count"] = uncovered_count
        subdomain["u3d_partition_global_overlap_count"] = overlap_count
        subdomain["u3d_partition_global_max_overlap"] = max_overlap

    # Compact diagnostics for quick sanity checks of the decomposition.
    if print_summary and subdomains:
        n_subdomains = len(subdomains)
        cell_counts = np.array([sd["cell_counts"] for sd in subdomains], dtype=int)
        local_err_norms_raw = np.array(
            [sd["u3d_partition_error_local_raw_norm_l2"] for sd in subdomains], dtype=float
        )
        local_err_rel_raw = np.array(
            [sd["u3d_partition_error_local_raw_rel_l2"] for sd in subdomains], dtype=float
        )
        rhs_norms = np.array([sd["partition_rhs_norm_l2"] for sd in subdomains], dtype=float)
        rhs_b0_norms = np.array([sd["partition_rhs_b0_norm_l2"] for sd in subdomains], dtype=float)
        rhs_cpl_norms = np.array(
            [sd["partition_rhs_coupling_norm_l2"] for sd in subdomains], dtype=float
        )
        skipped_q_counts = np.array(
            [sd["partition_skipped_q_vertices_count"] for sd in subdomains], dtype=int
        )
        active_q_counts = np.array(
            [sd["partition_active_q_vertices_count"] for sd in subdomains], dtype=int
        )
        zero_active_count = int(np.count_nonzero(active_q_counts == 0))
        print("Decomposition summary:")
        print(f"  subdomains: {n_subdomains}")
        p_nnz = np.array([sd["P"].nnz for sd in subdomains], dtype=int)
        print(
            "  P^Gamma nnz (= selected interface dofs, artificial cuts only): "
            f"min={p_nnz.min()} max={p_nnz.max()}"
        )
        print(
            "  cell_counts per axis: "
            f"min=({cell_counts[:, 0].min()}, {cell_counts[:, 1].min()}, {cell_counts[:, 2].min()}) "
            f"max=({cell_counts[:, 0].max()}, {cell_counts[:, 1].max()}, {cell_counts[:, 2].max()})"
        )
        print(
            "  ||u_full_sub - u_partition_sub_raw||_2: "
            f"min={local_err_norms_raw.min():.3e} "
            f"max={local_err_norms_raw.max():.3e} "
            f"avg={local_err_norms_raw.mean():.3e}"
        )
        print(
            "  relative ||u_full_sub - u_partition_sub_raw||_2 / ||u_full_sub||_2: "
            f"min={local_err_rel_raw.min():.3e} "
            f"max={local_err_rel_raw.max():.3e} "
            f"avg={local_err_rel_raw.mean():.3e}"
        )
        print(
            "  partition rhs norm ||rhs||_2: "
            f"min={rhs_norms.min():.3e} max={rhs_norms.max():.3e} avg={rhs_norms.mean():.3e}"
        )
        print(
            "  rhs components ||b0||_2 / ||coupling||_2 (avg): "
            f"{rhs_b0_norms.mean():.3e} / {rhs_cpl_norms.mean():.3e}"
        )
        print(
            "  skipped graph vertices in C (per subdomain): "
            f"min={skipped_q_counts.min()} max={skipped_q_counts.max()} avg={skipped_q_counts.mean():.1f}"
        )
        print(
            "  active graph vertices used for C (per subdomain): "
            f"min={active_q_counts.min()} max={active_q_counts.max()} avg={active_q_counts.mean():.1f} "
            f"zero_active={zero_active_count}"
        )
        ext_counts = np.array(
            [sd.get("exterior_dofs_eliminated", 0) for sd in subdomains], dtype=float
        )
        int_counts = np.array(
            [sd.get("interior_dofs_kept", 0) for sd in subdomains], dtype=float
        )
        tot_counts = np.maximum(ext_counts + int_counts, 1.0)
        ext_frac = ext_counts / tot_counts
        print(
            "  exterior dofs eliminated (per subdomain): "
            f"min={int(ext_counts.min())} max={int(ext_counts.max())} avg={ext_counts.mean():.1f}"
        )
        print(
            "  exterior dof fraction (per subdomain): "
            f"min={ext_frac.min():.3f} max={ext_frac.max():.3f} avg={ext_frac.mean():.3f}"
        )
        print(
            "  ||u_full - u_reconstructed_raw_partitions||_2: "
            f"abs={full_error_raw_l2:.3e} rel={full_error_raw_rel_l2:.3e}"
        )
        print(
            "  reconstructed full-domain coverage: "
            f"covered={covered_count}/{n_global} "
            f"uncovered={uncovered_count} "
            f"overlap_dofs={overlap_count} "
            f"max_overlap={max_overlap}"
        )

        top_k = min(5, n_subdomains)
        worst_ids = np.argsort(-local_err_rel_raw)[:top_k]
        print("  worst subdomains by relative local error (raw):")
        for wid in worst_ids:
            sd = subdomains[int(wid)]
            print(
                f"    {sd['ijk']}: "
                f"abs_err={sd['u3d_partition_error_local_raw_norm_l2']:.3e} "
                f"rel_err={sd['u3d_partition_error_local_raw_rel_l2']:.3e} "
                f"||rhs||={sd['partition_rhs_norm_l2']:.3e} "
                f"||b0||={sd['partition_rhs_b0_norm_l2']:.3e} "
                f"||cpl||={sd['partition_rhs_coupling_norm_l2']:.3e} "
                f"skipped_q={sd['partition_skipped_q_vertices_count']}"
            )

    return subdomains



def decomposeDomain(
    solver,
    boundary,
    x_ROM_lenght=5.0,
    y_ROM_lenght=5.0,
    z_ROM_lenght=5.0,
    restrict_global_C=False,
):
    """Build the subdomains and solve each one in isolation.

    This module ends where the local solve ends. The returned subdomains carry
    everything a transmission scheme needs -- partition_solver,
    local_to_global_dof, P, ijk, sol_3d_sub and the raw per-box errors -- and
    the schemes themselves live elsewhere:

        Robin_residual.apply_robin_residual  -- eqs. (4)-(5), one-shot
        Solve_schwarz_robin.solve_schwarz_robin -- iterative trace exchange
    """
    V, Q = solver.W
    meshV = V.mesh()
    meshQ = Q.mesh()
    sol_3d = solver.u3d
    sol_1d = solver.u1d
    """
    Decompose the 3D-1D solution into subdomains for ROM training.

    Parameters
    ----------
    meshV : dolfinx.mesh.Mesh
        The 3D mesh.
    meshQ : dolfinx.mesh.Mesh
        The 1D mesh.
    sol_3d : dolfinx.fem.Function
        The solved 3D field.
    sol_1d : dolfinx.fem.Function
        The solved 1D field.
    boundary : Boundary
        The boundary object defining the domain.

    Returns
    -------
    subdomains : list of dict
        A list of dictionaries, each containing the subdomain meshes and solutions.
    """
    # Placeholder for actual decomposition logic.
    # This returns a single full-domain entry with the mesh extents so the
    # caller can already consume the geometry information.
    coords = meshV.coordinates()
    x_min_global = float(coords[:, 0].min())
    x_max_global = float(coords[:, 0].max())
    y_min_global = float(coords[:, 1].min())
    y_max_global = float(coords[:, 1].max())
    z_min_global = float(coords[:, 2].min())
    z_max_global = float(coords[:, 2].max())

    dim_x_dir = x_max_global - x_min_global
    dim_y_dir = y_max_global - y_min_global
    dim_z_dir = z_max_global - z_min_global

    unique_x = np.unique(coords[:, 0])
    unique_y = np.unique(coords[:, 1])
    unique_z = np.unique(coords[:, 2])
    dx = float(np.min(np.diff(unique_x))) if len(unique_x) > 1 else 0.0
    dy = float(np.min(np.diff(unique_y))) if len(unique_y) > 1 else 0.0
    dz = float(np.min(np.diff(unique_z))) if len(unique_z) > 1 else 0.0

    subdomains = []

    def axis_index(axis_values, value, atol=1e-12):
        idx = int(np.argmin(np.abs(axis_values - value)))
        if not np.isclose(axis_values[idx], value, atol=atol, rtol=0.0):
            raise RuntimeError(
                f"Point coordinate {value:.16e} is not aligned with mesh grid."
            )
        return idx

    # Structured grid index triplets present in the original full 3D mesh.
    mesh_triplets = set()
    for c in meshV.coordinates():
        ix = axis_index(unique_x, c[0])
        iy = axis_index(unique_y, c[1])
        iz = axis_index(unique_z, c[2])
        mesh_triplets.add((ix, iy, iz))

    # Map full-domain 3D dof coordinates to structured grid indices.
    V_coords = V.tabulate_dof_coordinates().reshape((V.dim(), -1))
    triplet_to_global_dof = {}
    for dof, c in enumerate(V_coords):
        ix = axis_index(unique_x, c[0])
        iy = axis_index(unique_y, c[1])
        iz = axis_index(unique_z, c[2])
        triplet_to_global_dof[(ix, iy, iz)] = int(dof)



    num_x_subdomains = max(1, math.ceil(dim_x_dir / x_ROM_lenght))
    num_y_subdomains = max(1, math.ceil(dim_y_dir / y_ROM_lenght))
    num_z_subdomains = max(1, math.ceil(dim_z_dir / z_ROM_lenght))



    for i in range(num_x_subdomains):
        for j in range(num_y_subdomains):
            for k in range(num_z_subdomains):
                # Define the subdomain extents
                x_target_min = x_min_global + i * x_ROM_lenght
                x_target_max = min(x_target_min + x_ROM_lenght, x_max_global)
                y_target_min = y_min_global + j * y_ROM_lenght
                y_target_max = min(y_target_min + y_ROM_lenght, y_max_global)
                z_target_min = z_min_global + k * z_ROM_lenght
                z_target_max = min(z_target_min + z_ROM_lenght, z_max_global)

                x_start = int(np.searchsorted(unique_x, x_target_min, side="left"))
                x_stop = int(np.searchsorted(unique_x, x_target_max, side="right") - 1)
                y_start = int(np.searchsorted(unique_y, y_target_min, side="left"))
                y_stop = int(np.searchsorted(unique_y, y_target_max, side="right") - 1)
                z_start = int(np.searchsorted(unique_z, z_target_min, side="left"))
                z_stop = int(np.searchsorted(unique_z, z_target_max, side="right") - 1)

                x_start = max(0, min(x_start, len(unique_x) - 1))
                x_stop = max(x_start, min(x_stop, len(unique_x) - 1))
                y_start = max(0, min(y_start, len(unique_y) - 1))
                y_stop = max(y_start, min(y_stop, len(unique_y) - 1))
                z_start = max(0, min(z_start, len(unique_z) - 1))
                z_stop = max(z_start, min(z_stop, len(unique_z) - 1))

                x_min = float(unique_x[x_start])
                x_max = float(unique_x[x_stop])
                y_min = float(unique_y[y_start])
                y_max = float(unique_y[y_stop])
                z_min = float(unique_z[z_start])
                z_max = float(unique_z[z_stop])

                nx = max(1, x_stop - x_start)
                ny = max(1, y_stop - y_start)
                nz = max(1, z_stop - z_start)

                meshV_sub = BoxMesh(
                    Point(x_min, y_min, z_min),
                    Point(x_max, y_max, z_max),
                    nx, ny, nz,
                )

                # Mark cells with the SAME criterion as Solver3D1D: default 111
                # (exterior), 222 only where the cell midpoint is inside the
                # analytic boundary. Marking every cell 222 makes each box solve
                # the PDE across regions that lie outside the physical domain,
                # where the full-domain reference is identically zero.
                V_cell_markers_sub = MeshFunction("size_t", meshV_sub, 3, 111)
                inside_count_sub = 0
                for cell in cells(meshV_sub):
                    mp = cell.midpoint()
                    if boundary is None or boundary([mp.x(), mp.y(), mp.z()]):
                        V_cell_markers_sub[cell] = 222
                        inside_count_sub += 1

                V_sub = FunctionSpace(meshV_sub, "CG", 1)
                sol_3d_sub = Function(V_sub)
                sub_coords_local = V_sub.tabulate_dof_coordinates().reshape((V_sub.dim(), -1))
                local_min = sub_coords_local.min(axis=0)
                local_max = sub_coords_local.max(axis=0)
                local_span = np.where(local_max > local_min, local_max - local_min, 1.0)
                target_min = np.array([x_min, y_min, z_min], dtype=float)
                target_max = np.array([x_max, y_max, z_max], dtype=float)
                sub_coords = target_min + (sub_coords_local - local_min) * (
                    (target_max - target_min) / local_span
                )
                sol_3d_sub.vector()[:] = np.array(
                    [sol_3d(Point(*xyz)) for xyz in sub_coords],
                    dtype=float,
                )

                local_to_global_dof = np.array(
                    [
                        triplet_to_global_dof[
                            (
                                axis_index(unique_x, xyz[0], atol=1e-10),
                                axis_index(unique_y, xyz[1], atol=1e-10),
                                axis_index(unique_z, xyz[2], atol=1e-10),
                            )
                        ]
                        for xyz in sub_coords
                    ],
                    dtype=int,
                )

                subdomains.append({
                    "ijk": (i, j, k),
                    "meshV": meshV_sub,
                    "V_cell_markers": V_cell_markers_sub,
                    "V_sub": V_sub,
                    "meshQ": meshQ,
                    "sol_3d": sol_3d,
                    "sol_3d_sub": sol_3d_sub,
                    "sol_1d": sol_1d,
                    "boundary": boundary,
                    "dim_x_dir": x_max - x_min,
                    "dim_y_dir": y_max - y_min,
                    "dim_z_dir": z_max - z_min,
                    "cell_counts": (nx, ny, nz),
                    "x_start": x_start,
                    "x_stop": x_stop,
                    "y_start": y_start,
                    "y_stop": y_stop,
                    "z_start": z_start,
                    "z_stop": z_stop,
                    "local_to_global_dof": local_to_global_dof,
                })


    for idx, subdomain in enumerate(subdomains):
        x_start = int(subdomain["x_start"])
        x_stop = int(subdomain["x_stop"])
        y_start = int(subdomain["y_start"])
        y_stop = int(subdomain["y_stop"])
        z_start = int(subdomain["z_start"])
        z_stop = int(subdomain["z_stop"])

        # P^Gamma is a RESTRICTION: it keeps the rows of the global residual
        # that sit on this box's ARTIFICIAL cut planes and discards every other
        # row. It is a projector -- 1 on the diagonal for selected dofs, zero
        # everywhere else -- so (P r)[d] = r[d] on the interface and 0 off it.
        #
        # It must NOT be a neighbour stencil: summing r over a 27-point cube
        # multiplies the extracted flux by ~27 and makes the local solves blow
        # up. Coupling between neighbouring interface nodes is the job of the
        # interface mass matrix G^Gamma, not of the restriction.
        #
        # Faces lying on the true outer boundary are excluded: they are not
        # interfaces between subdomains and carry no transmission data.
        x_lo_cut = x_start > 0
        x_hi_cut = x_stop < len(unique_x) - 1
        y_lo_cut = y_start > 0
        y_hi_cut = y_stop < len(unique_y) - 1
        z_lo_cut = z_start > 0
        z_hi_cut = z_stop < len(unique_z) - 1

        border_triplets = set()

        # z-faces
        for ix in range(x_start, x_stop + 1):
            for iy in range(y_start, y_stop + 1):
                if z_lo_cut:
                    border_triplets.add((ix, iy, z_start))
                if z_hi_cut:
                    border_triplets.add((ix, iy, z_stop))

        # x-faces
        for iy in range(y_start, y_stop + 1):
            for iz in range(z_start, z_stop + 1):
                if x_lo_cut:
                    border_triplets.add((x_start, iy, iz))
                if x_hi_cut:
                    border_triplets.add((x_stop, iy, iz))

        # y-faces
        for ix in range(x_start, x_stop + 1):
            for iz in range(z_start, z_stop + 1):
                if y_lo_cut:
                    border_triplets.add((ix, y_start, iz))
                if y_hi_cut:
                    border_triplets.add((ix, y_stop, iz))

        p_rows = []
        p_cols = []
        p_data = []

        for ix, iy, iz in border_triplets:
            assert (ix, iy, iz) in mesh_triplets, (
                f"Border index ({ix}, {iy}, {iz}) is missing from the original mesh."
            )
            key = (ix, iy, iz)
            if key in triplet_to_global_dof:
                d = triplet_to_global_dof[key]
                p_rows.append(d)
                p_cols.append(d)
                p_data.append(1.0)

        P = csr_matrix(
            (p_data, (p_rows, p_cols)),
            shape=(V.dim(), V.dim()),
        )
        subdomain["P"] = P
        subdomain["n_interface_dofs"] = int(len(p_rows))

    return solve_partition_domain(
        subdomains, solver, print_summary=True,
        restrict_global_C=restrict_global_C,
    )





if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Build a simple analytic (non-OpenCCO) vascular domain and "
                    "solve the 3D-1D problem with the no-penalty solver.",
    )
    parser.add_argument("-name",   type=str, required=True,
                        help="subfolder name inside nets/ (e.g. sphere01)")
    parser.add_argument("-inlet",  type=int, default=10,
                        help="number of inflow vessels (n_vasi)")
    parser.add_argument("-outlet", type=int, default=10,
                        help="ramifications per vessel (n_ramifications)")
    parser.add_argument("-n",      type=int, default=50,
                        help="3D background mesh resolution")
    parser.add_argument("-sigma1d", type=float, default=1.0,
                        help="1D conductivity (sigma1d)")
    parser.add_argument("-sigma3d", type=float, default=1e-3,
                        help="3D conductivity (sigma3d)")
    parser.add_argument("-kappa", type=float, default=1.0,
                        help="coupling coefficient (kappa)")
    parser.add_argument("-radius", type=float, default=5.0,
                        help="radius of the spherical boundary")
    args = parser.parse_args()

    # nets/{name}/ holds every mesh file this run writes and the solver reads.
    #
    # Domain's export_* methods build filenames as f"{self.name}_marked_mesh.xdmf"
    # (i.e. they add the trailing '_'), while the solver reads them back as
    # f"{path_to_1D_mesh}marked_mesh.xdmf" (no added '_'). So self.name must NOT
    # end in '_', and path_to_1D_mesh MUST — otherwise the underscores don't line
    # up and the solver looks for a file the domain never wrote.
    net_dir = os.path.join("nets", args.name)
    os.makedirs(net_dir, exist_ok=True)
    name_stem = os.path.join(net_dir, args.name)   # nets/NAME/NAME      (no trailing _)
    mesh_prefix = f"{name_stem}_"                   # nets/NAME/NAME_     (solver prefix)

    

    boundary = SphereBoundary(
        radius=args.radius,
        inlet_points=random_sphere_points(
            40,
            x_sign=-1,
            min_x=0.2,
            min_dist_to_boundary=0.06,
            radius=args.radius,
        ),
        outlet_points=random_sphere_points(
            40,
            x_sign=+1,
            min_x=0.2,
            min_dist_to_boundary=0.06,
            radius=args.radius,
        ),
        border_eps=10e-1,
    )

    check_sphere_domain_consistency(
        boundary=boundary,
        n_min=-args.radius,
        n_max=args.radius,
    )

    # --- 1. build the analytic-boundary domain (unit sphere from Boundary.py) ---
    domain = Domain(
        name            = name_stem,
        n_vasi          = args.inlet,
        n_ramifications = args.outlet,
        boundary        = boundary,
        n_min = -args.radius, n_max = args.radius
    ).build()

    domain.export_box()
    domain.export_reticolo()
    domain.export_vaso()
    domain.export_xdmf()

    # --- 2. run the solver WITHOUT the boundary penalty ---
    # exterior='dirichlet' eliminates the exterior DOFs exactly instead of
    # pinning them with a penalty term, so the interface is not contaminated.
    out_dir = (
        f"./solution/Simple{args.name}_n{args.n}"
        f"_s1d{args.sigma1d}_s3d{args.sigma3d}_k{args.kappa}"
    )
    solver  = Solver3D1D(
        path_to_1D_mesh = mesh_prefix,
        boundary        = boundary,
        n               = args.n,
        sigma3d         = args.sigma3d,
        sigma1d         = args.sigma1d,
        kappa           = args.kappa,
        exterior        = "dirichlet",
    ).build().solve()

    solver.save(out_dir)
    solver.save_paraview(f"{out_dir}/paraview")



    subdomains = decomposeDomain(solver, boundary)