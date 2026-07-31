# =============================================================================
# Entry point
# =============================================================================

import math
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import issparse
from scipy.sparse.linalg import spsolve, splu

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
from xii import ii_convert

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


def compute_robin(solver, rho_robin=1.0):
    """Compute full-mesh Robin residual ingredients and rhs vector."""
    V, _ = solver.W

    # Full-mesh 3D blocks and vectors used by the Robin residual formula.
    A3 = matrix_to_csr(solver.A[0][0])

    # G is the plain 3D volume mass matrix on V (not the 3D-1D coupling block).
    u_mass, v_mass = TrialFunction(V), TestFunction(V)
    G3_dolfin = assemble(inner(u_mass, v_mass) * solver.d_omega(222))
    G3 = matrix_to_csr(G3_dolfin)

    b0 = solver.b[0]
    if isinstance(b0, np.ndarray):
        b3 = b0.astype(float, copy=True).ravel()
    elif hasattr(b0, "get_local"):
        b3 = b0.get_local()
    else:
        b3 = ii_convert(b0).get_local()
    u3_star = solver.u3d.vector().get_local()

    robin_rhs = b3 - A3.dot(u3_star) + float(rho_robin) * G3.dot(u3_star)
    return {
        "A3": A3,
        "G3": G3,
        "b3": b3,
        "u3_star": u3_star,
        "robin_rhs": robin_rhs,
    }


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
    Assemble the Robin interface mass matrix M_gamma = \\int_{Gamma_art} u v ds
    on the tagged artificial facets of one subdomain.

    This is a SURFACE integral on the cut planes -- not a volume mass matrix.
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
    M_dolfin = assemble(inner(u, v) * ds_art(ARTIFICIAL_FACET_TAG))
    return matrix_to_csr(M_dolfin)


def eliminate_exterior_local(partition_solver, interior_tag=222):
    """
    Apply the same exterior-dof elimination that Solver3D1D does, to one
    subdomain system, in place.

    A dof is interior if it belongs to at least one cell marked `interior_tag`.
    Every other dof lies fully outside the physical domain: its row/column is
    dropped, replaced by a unit diagonal, and its rhs zeroed, pinning u=0 there
    exactly as the full-domain solve does.

    Returns (n_ext, n_int).
    """
    V = partition_solver.V
    markers = partition_solver.meshV_markers
    mesh = partition_solver.meshV
    dm = V.dofmap()
    n = V.dim()

    has_int = np.zeros(n, dtype=bool)
    for cell in cells(mesh):
        if markers[cell] == interior_tag:
            has_int[dm.cell_dofs(cell.index())] = True

    ext_dofs = np.flatnonzero(~has_int)
    int_dofs = np.flatnonzero(has_int)

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


def solve_schwarz_robin(
    subdomains,
    solver,
    rho_robin=None,
    max_iter=30,
    tol=1e-6,
    print_summary=True,
):
    """
    Additive Schwarz iteration with a Robin transmission condition on the
    artificial cut planes.

    Per subdomain the system is

        (A_local + rho * M_gamma) u^(k+1) = rhs_local + rho * M_gamma u_nbr^(k)

    The Robin term is added to the matrix ONCE: A_local + rho*M_gamma is
    assembled and LU-factorized a single time, outside the loop. Only the
    right-hand side changes between sweeps, refreshed from the neighbours'
    current interface traces. This is why iterating reduces the error -- on the
    first sweep u_nbr^(0) = 0 carries no information, and coupling between boxes
    only propagates on later sweeps.

    Uses the previous iterate's trace, never the ground-truth field, so this is
    a scheme that would work with an unknown target.
    """
    V_global = solver.W[0]
    n_global = V_global.dim()

    if rho_robin is None:
        # Standard Robin scaling rho ~ sigma3d / h.
        hmax = solver.meshV.hmax() if hasattr(solver, "meshV") else 1.0
        rho_robin = float(solver.sigma3d) / max(float(hmax), 1e-30)

    coords_global = V_global.tabulate_dof_coordinates().reshape((n_global, -1))
    g_min = coords_global.min(axis=0)
    g_max = coords_global.max(axis=0)

    # ---- one-time setup: Robin term into the matrix, then factorize ----------
    states = []
    for subdomain in subdomains:
        ps = subdomain["partition_solver"]
        facet_markers, n_tagged = mark_artificial_facets(ps, g_min, g_max)
        M_gamma = assemble_robin_interface(ps, facet_markers)

        # Exterior dofs stay pinned to u=0: drop their Robin rows/cols so the
        # transmission condition never reactivates them.
        ext_dofs = getattr(ps, "ext_dofs", None)
        if ext_dofs is not None and np.size(ext_dofs):
            keep = np.ones(M_gamma.shape[0], dtype=bool)
            keep[np.asarray(ext_dofs, dtype=int)] = False
            D = csr_matrix(
                (keep.astype(float), (np.arange(keep.size), np.arange(keep.size))),
                shape=M_gamma.shape,
            )
            M_gamma = D @ M_gamma @ D

        A_robin = (ps.A + rho_robin * M_gamma).tocsc()
        states.append({
            "subdomain": subdomain,
            "ps": ps,
            "M_gamma": M_gamma,
            "lu": splu(A_robin),
            "rhs_local": np.array(ps.rhs, dtype=float),
            "l2g": subdomain["local_to_global_dof"],
            "n_tagged_facets": n_tagged,
            "u": np.zeros(ps.V.dim(), dtype=float),
        })
        subdomain["schwarz_n_artificial_facets"] = n_tagged

    # ---- iterate: only the rhs changes --------------------------------------
    u3d_full_vec = solver.u3d.vector().get_local()
    full_ref_l2 = float(np.linalg.norm(u3d_full_vec))
    history = []

    for it in range(1, max_iter + 1):
        # Global trace field from the current iterates (averaged on overlaps).
        acc = np.zeros(n_global, dtype=float)
        cnt = np.zeros(n_global, dtype=float)
        for st in states:
            acc[st["l2g"]] += st["u"]
            cnt[st["l2g"]] += 1.0
        u_glob = np.zeros(n_global, dtype=float)
        nz = cnt > 0.0
        u_glob[nz] = acc[nz] / cnt[nz]

        delta = 0.0
        for st in states:
            u_nbr = u_glob[st["l2g"]]
            rhs = st["rhs_local"] + rho_robin * (st["M_gamma"] @ u_nbr)
            u_new = st["lu"].solve(rhs)
            delta = max(delta, float(np.linalg.norm(u_new - st["u"])))
            st["u_next"] = u_new
        for st in states:
            st["u"] = st["u_next"]

        # Reconstruct and measure against the full-domain reference.
        acc = np.zeros(n_global, dtype=float)
        cnt = np.zeros(n_global, dtype=float)
        for st in states:
            acc[st["l2g"]] += st["u"]
            cnt[st["l2g"]] += 1.0
        rec = np.zeros(n_global, dtype=float)
        nz = cnt > 0.0
        rec[nz] = acc[nz] / cnt[nz]
        rec[~nz] = u3d_full_vec[~nz]

        rel_glob = (
            float(np.linalg.norm(u3d_full_vec - rec)) / full_ref_l2
            if full_ref_l2 > 0.0
            else 0.0
        )
        rel_loc = []
        for st in states:
            uf = st["subdomain"]["sol_3d_sub"].vector().get_local()
            nf = float(np.linalg.norm(uf))
            rel_loc.append(
                float(np.linalg.norm(uf - st["u"])) / nf if nf > 1e-12 else 0.0
            )
        rel_loc_avg = float(np.mean(rel_loc))
        history.append((it, delta, rel_loc_avg, rel_glob))

        if delta < tol:
            break

    for st in states:
        sd = st["subdomain"]
        fn = Function(st["ps"].V)
        fn.vector()[:] = st["u"]
        sd["u3d_partition_schwarz"] = fn
        uf = sd["sol_3d_sub"].vector().get_local()
        nf = float(np.linalg.norm(uf))
        sd["u3d_partition_error_schwarz_norm_l2"] = float(np.linalg.norm(uf - st["u"]))
        sd["u3d_partition_error_schwarz_rel_l2"] = (
            sd["u3d_partition_error_schwarz_norm_l2"] / nf if nf > 1e-12 else 0.0
        )

    if print_summary and history:
        n_facets = np.array([st["n_tagged_facets"] for st in states], dtype=float)
        print("")
        print("Additive Schwarz / Robin iteration:")
        print(
            f"  rho_robin={rho_robin:.4e}  max_iter={max_iter}  tol={tol:.1e}  "
            f"artificial facets/subdomain: min={int(n_facets.min())} "
            f"max={int(n_facets.max())}"
        )
        print("  iter |   delta    | avg rel local | rel global")
        for it, delta, rl, rg in history:
            print(f"  {it:4d} | {delta:.4e} | {rl:.6e}  | {rg:.6e}")
        final_loc = history[-1][2]
        final_glob = history[-1][3]
        raw_loc = float(
            np.mean([
                sd["u3d_partition_error_local_raw_rel_l2"] for sd in subdomains
            ])
        )
        print(
            f"  raw (no transmission) avg rel local: {raw_loc:.6e}"
            f"  ->  Schwarz: {final_loc:.6e}"
        )
        if final_loc > 1e-14:
            print(f"  improvement over raw: {raw_loc / final_loc:.2f}x")
        print(f"  final reconstructed global rel error: {final_glob:.6e}")

    return history


def diagnose_exact_dirichlet(subdomains, solver, print_summary=True):
    """
    Step-1 diagnostic: re-solve each subdomain with EXACT Dirichlet data taken
    from the full-domain solution on the artificial cut planes.

    This is an offline upper-bound check, not a usable scheme: it consumes the
    ground-truth field `solver.u3d`, which is unknown in production. Its only
    purpose is to answer one question -- is the ~40% local error caused by the
    missing inter-subdomain transmission condition, or by something else?

    Interpretation
    --------------
    err_dirichlet << err_raw  ->  the error IS the missing transmission
        condition. A converged Schwarz/Robin iteration can recover roughly
        this much, and `err_dirichlet` is the floor it converges to.
    err_dirichlet ~= err_raw  ->  the error is NOT (only) the interface. Look
        instead at the coupling operator C, the skipped graph vertices, or the
        local operator itself. Building a Schwarz loop would be wasted effort.

    Only faces on an artificial cut are constrained. Faces that lie on the
    global mesh boundary are left with whatever natural condition the local
    form implies, matching how the full-domain solve treats them.
    """
    V_global = solver.W[0]
    u3_star = solver.u3d.vector().get_local()

    coords_global = V_global.tabulate_dof_coordinates().reshape((V_global.dim(), -1))
    g_min = coords_global.min(axis=0)
    g_max = coords_global.max(axis=0)

    results = []

    for subdomain in subdomains:
        V_sub = subdomain["V_sub"]
        local_to_global_dof = subdomain["local_to_global_dof"]
        sub_coords = V_sub.tabulate_dof_coordinates().reshape((V_sub.dim(), -1))

        # Guard: the local reference is built by point-evaluation while
        # local_to_global_dof is an exact index map. If these disagree, every
        # error number below is polluted, so fail loudly instead of reporting.
        u_full_local = subdomain["sol_3d_sub"].vector().get_local()
        u_mapped = u3_star[local_to_global_dof]
        map_mismatch = float(np.max(np.abs(u_full_local - u_mapped)))
        ref_scale = max(float(np.max(np.abs(u_full_local))), 1e-30)
        if map_mismatch / ref_scale > 1e-8:
            raise RuntimeError(
                f"Subdomain {subdomain['ijk']}: sol_3d_sub disagrees with "
                f"u3_star[local_to_global_dof] (max abs diff {map_mismatch:.3e}, "
                f"rel {map_mismatch / ref_scale:.3e}). The dof map or the "
                "point-evaluation reference is wrong; diagnostic aborted."
            )

        # Reuse the exact same local operator/rhs as the raw path, so the only
        # difference between err_raw and err_dirichlet is the interface data.
        partition_solver = subdomain["partition_solver"]
        A_local = partition_solver.A.tolil(copy=True)
        rhs_local = np.array(partition_solver.rhs, dtype=float)

        # Identify artificial-cut boundary dofs: on a face of the local box,
        # but not on the corresponding face of the global mesh.
        tol = 1e-9
        sub_lo = sub_coords.min(axis=0)
        sub_hi = sub_coords.max(axis=0)
        on_artificial = np.zeros(V_sub.dim(), dtype=bool)
        for axis in range(3):
            at_lo = np.abs(sub_coords[:, axis] - sub_lo[axis]) < tol
            at_hi = np.abs(sub_coords[:, axis] - sub_hi[axis]) < tol
            if abs(sub_lo[axis] - g_min[axis]) > tol:
                on_artificial |= at_lo
            if abs(sub_hi[axis] - g_max[axis]) > tol:
                on_artificial |= at_hi

        # Never constrain an eliminated exterior dof: it is already pinned to
        # u=0 by the elimination, and overwriting its row would reintroduce the
        # PDE outside the physical domain.
        ext_dofs = getattr(partition_solver, "ext_dofs", None)
        if ext_dofs is not None and np.size(ext_dofs):
            on_artificial[np.asarray(ext_dofs, dtype=int)] = False

        dirichlet_dofs = np.where(on_artificial)[0]

        # Impose u = u_star on those dofs by row replacement.
        for d in dirichlet_dofs:
            A_local.rows[d] = [int(d)]
            A_local.data[d] = [1.0]
            rhs_local[d] = u_mapped[d]

        u_dirichlet = spsolve(A_local.tocsr(), rhs_local)

        err_dirichlet = u_full_local - u_dirichlet
        err_dirichlet_l2 = float(np.linalg.norm(err_dirichlet))
        full_local_norm = float(np.linalg.norm(u_full_local))
        rel_dirichlet = err_dirichlet_l2 / full_local_norm if full_local_norm > 1e-12 else 0.0

        u_dirichlet_fn = Function(V_sub)
        u_dirichlet_fn.vector()[:] = u_dirichlet

        subdomain["u3d_partition_dirichlet"] = u_dirichlet_fn
        subdomain["u3d_partition_error_dirichlet_norm_l2"] = err_dirichlet_l2
        subdomain["u3d_partition_error_dirichlet_rel_l2"] = rel_dirichlet
        subdomain["dirichlet_dof_count"] = int(dirichlet_dofs.size)
        subdomain["dirichlet_interior_dof_count"] = int(V_sub.dim() - dirichlet_dofs.size)

        results.append({
            "ijk": subdomain["ijk"],
            "rel_raw": subdomain["u3d_partition_error_local_raw_rel_l2"],
            "rel_dirichlet": rel_dirichlet,
            "n_dirichlet": int(dirichlet_dofs.size),
        })

    if print_summary and results:
        rel_raw = np.array([r["rel_raw"] for r in results], dtype=float)
        rel_dir = np.array([r["rel_dirichlet"] for r in results], dtype=float)
        print("")
        print("Step-1 diagnostic (exact Dirichlet on artificial cut planes):")
        print(
            "  relative local error, raw (Neumann cuts):      "
            f"min={rel_raw.min():.3e} max={rel_raw.max():.3e} avg={rel_raw.mean():.3e}"
        )
        print(
            "  relative local error, exact-Dirichlet cuts:    "
            f"min={rel_dir.min():.3e} max={rel_dir.max():.3e} avg={rel_dir.mean():.3e}"
        )
        if rel_dir.mean() > 1e-14:
            print(f"  error reduction factor (avg): {rel_raw.mean() / rel_dir.mean():.1f}x")
        print("  per-subdomain (ijk: raw -> dirichlet, #constrained dofs):")
        for r in sorted(results, key=lambda x: -x["rel_raw"]):
            print(
                f"    {r['ijk']}: {r['rel_raw']:.3e} -> {r['rel_dirichlet']:.3e} "
                f"({r['n_dirichlet']} dofs)"
            )

        # Judge on the reduction factor, not an absolute floor: the question is
        # whether the interface is the dominant remaining error, not whether the
        # floor has reached discretization error (other defects also raise it).
        verdict_floor = rel_dir.mean()
        reduction = rel_raw.mean() / verdict_floor if verdict_floor > 1e-14 else np.inf
        if reduction >= 3.0:
            print(
                "  VERDICT: interface data dominates the remaining error "
                f"({reduction:.1f}x reduction). A converged Schwarz/Robin iteration "
                f"targets ~{verdict_floor:.1%} relative error. The floor itself is "
                "set by the other error sources (coupling operator C, skipped graph "
                "vertices, discretization) and bounds what any transmission "
                "condition can achieve."
            )
        elif reduction >= 1.5:
            print(
                f"  VERDICT: mixed ({reduction:.1f}x reduction). The interface matters "
                "but is not dominant; expect only partial gains from a Schwarz loop "
                "until the floor is lowered."
            )
        else:
            print(
                "  VERDICT: exact interface data does NOT recover the solution "
                f"({reduction:.1f}x reduction). The dominant error is elsewhere "
                "(coupling operator C, skipped graph vertices, or the local "
                "operator). Do not build the Schwarz loop yet."
            )

    return results


def solve_partition_domain(subdomains, solver, rho_robin=1.0, print_summary=True):
    """Apply full-mesh Robin rhs to each partition and attach diagnostics."""
    robin_data = compute_robin(solver, rho_robin=rho_robin)
    A3 = robin_data["A3"]
    G3 = robin_data["G3"]
    b3 = robin_data["b3"]
    u3_star = robin_data["u3_star"]
    robin_rhs = robin_data["robin_rhs"]

    V_global = solver.W[0]
    n_global = V_global.dim()
    u_partition_sum_raw = np.zeros(n_global, dtype=float)
    u_partition_count_raw = np.zeros(n_global, dtype=float)
    u_partition_sum = np.zeros(n_global, dtype=float)
    u_partition_count = np.zeros(n_global, dtype=float)

    for subdomain in subdomains:
        f_star_minus = subdomain["P"].dot(robin_rhs)
        local_to_global_dof = subdomain["local_to_global_dof"]
        f_star_minus_local = np.asarray(f_star_minus[local_to_global_dof], dtype=float)
        f_star_minus_local_fn = Function(subdomain["V_sub"])
        f_star_minus_local_fn.vector()[:] = f_star_minus_local

        subdomain["A3"] = A3
        subdomain["G3"] = G3
        subdomain["b3"] = b3
        subdomain["u3_star"] = u3_star
        subdomain["f_star_minus"] = f_star_minus
        subdomain["f_star_minus_norm_l2"] = float(np.linalg.norm(f_star_minus))
        subdomain["f_star_minus_local"] = f_star_minus_local
        subdomain["f_star_minus_local_fn"] = f_star_minus_local_fn
        subdomain["f_star_minus_local_norm_l2"] = float(np.linalg.norm(f_star_minus_local))

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
        ).build()

        # Pin u=0 outside the physical domain, as the full-domain solve does.
        n_ext, n_int = eliminate_exterior_local(partition_solver, interior_tag=222)
        subdomain["exterior_dofs_eliminated"] = n_ext
        subdomain["interior_dofs_kept"] = n_int

        partition_solver.solve()
        subdomain["partition_solver"] = partition_solver

        # Add Robin flux term after solving the partition PDE.
        u_part_raw = partition_solver.u3d.vector().get_local()
        u_part_robin = u_part_raw + f_star_minus_local
        u3d_partition_post_robin = Function(subdomain["V_sub"])
        u3d_partition_post_robin.vector()[:] = u_part_robin

        subdomain["u3d_partition_raw"] = partition_solver.u3d
        subdomain["u3d_partition"] = u3d_partition_post_robin
        subdomain["u3d_partition_robin_correction_norm_l2"] = float(np.linalg.norm(f_star_minus_local))

        # Local reference-vs-partition error on the subdomain mesh.
        u_full_local = subdomain["sol_3d_sub"].vector().get_local()
        local_error_raw = u_full_local - u_part_raw
        u_part_local = u_part_robin
        local_error = u_full_local - u_part_local
        u_full_local_norm = float(np.linalg.norm(u_full_local))
        u_part_local_norm = float(np.linalg.norm(u_part_local))
        subdomain["u3d_partition_error_local_raw"] = local_error_raw
        subdomain["u3d_partition_error_local_raw_norm_l2"] = float(np.linalg.norm(local_error_raw))
        subdomain["u3d_partition_error_local_raw_rel_l2"] = (
            subdomain["u3d_partition_error_local_raw_norm_l2"] / u_full_local_norm
            if u_full_local_norm > 1e-12
            else 0.0
        )
        subdomain["u3d_partition_error_local"] = local_error
        subdomain["u3d_partition_error_local_norm_l2"] = float(np.linalg.norm(local_error))
        subdomain["u3d_full_local_norm_l2"] = u_full_local_norm
        subdomain["u3d_partition_local_norm_l2"] = u_part_local_norm
        subdomain["u3d_partition_error_local_rel_l2"] = (
            subdomain["u3d_partition_error_local_norm_l2"] / u_full_local_norm
            if u_full_local_norm > 1e-12
            else 0.0
        )
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

        # Contribute partition solution to a reconstructed full-domain field.
        u_partition_sum[local_to_global_dof] += u_part_local
        u_partition_count[local_to_global_dof] += 1.0

    covered_raw = u_partition_count_raw > 0.0
    u3d_reconstructed_raw_vec = np.zeros(n_global, dtype=float)
    u3d_reconstructed_raw_vec[covered_raw] = (
        u_partition_sum_raw[covered_raw] / u_partition_count_raw[covered_raw]
    )

    covered = u_partition_count > 0.0
    u3d_reconstructed_vec = np.zeros(n_global, dtype=float)
    u3d_reconstructed_vec[covered] = u_partition_sum[covered] / u_partition_count[covered]

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
    overlap_count = int(np.count_nonzero(u_partition_count > 1.0))
    max_overlap = int(u_partition_count.max()) if len(u_partition_count) else 0

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
        p_nnz = np.array([sd["P"].nnz for sd in subdomains], dtype=int)
        f_norms = np.array([sd["f_star_minus_norm_l2"] for sd in subdomains], dtype=float)
        local_err_norms_raw = np.array(
            [sd["u3d_partition_error_local_raw_norm_l2"] for sd in subdomains], dtype=float
        )
        local_err_norms = np.array(
            [sd["u3d_partition_error_local_norm_l2"] for sd in subdomains], dtype=float
        )
        local_err_rel_raw = np.array(
            [sd["u3d_partition_error_local_raw_rel_l2"] for sd in subdomains], dtype=float
        )
        local_err_rel = np.array(
            [sd["u3d_partition_error_local_rel_l2"] for sd in subdomains], dtype=float
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
        print(
            "  cell_counts per axis: "
            f"min=({cell_counts[:, 0].min()}, {cell_counts[:, 1].min()}, {cell_counts[:, 2].min()}) "
            f"max=({cell_counts[:, 0].max()}, {cell_counts[:, 1].max()}, {cell_counts[:, 2].max()})"
        )
        print(
            "  P.nnz: "
            f"min={p_nnz.min()} max={p_nnz.max()} avg={float(p_nnz.mean()):.1f}"
        )
        print(
            "  ||f_star_minus||_2: "
            f"min={f_norms.min():.3e} max={f_norms.max():.3e} avg={f_norms.mean():.3e}"
        )
        print(
            "  ||u_full_sub - u_partition_sub_raw||_2: "
            f"min={local_err_norms_raw.min():.3e} "
            f"max={local_err_norms_raw.max():.3e} "
            f"avg={local_err_norms_raw.mean():.3e}"
        )
        print(
            "  ||u_full_sub - u_partition_sub_post_robin||_2: "
            f"min={local_err_norms.min():.3e} "
            f"max={local_err_norms.max():.3e} "
            f"avg={local_err_norms.mean():.3e}"
        )
        print(
            "  relative ||u_full_sub - u_partition_sub_raw||_2 / ||u_full_sub||_2: "
            f"min={local_err_rel_raw.min():.3e} "
            f"max={local_err_rel_raw.max():.3e} "
            f"avg={local_err_rel_raw.mean():.3e}"
        )
        print(
            "  relative ||u_full_sub - u_partition_sub_post_robin||_2 / ||u_full_sub||_2: "
            f"min={local_err_rel.min():.3e} "
            f"max={local_err_rel.max():.3e} "
            f"avg={local_err_rel.mean():.3e}"
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
            "  ||u_full - u_reconstructed_post_robin_partitions||_2: "
            f"abs={full_error_l2:.3e} rel={full_error_rel_l2:.3e}"
        )
        print(
            "  reconstructed full-domain coverage: "
            f"covered={covered_count}/{n_global} "
            f"uncovered={uncovered_count} "
            f"overlap_dofs={overlap_count} "
            f"max_overlap={max_overlap}"
        )

        top_k = min(5, n_subdomains)
        worst_ids = np.argsort(-local_err_rel)[:top_k]
        print("  worst subdomains by relative local error:")
        for wid in worst_ids:
            sd = subdomains[int(wid)]
            print(
                f"    {sd['ijk']}: "
                f"abs_err={sd['u3d_partition_error_local_norm_l2']:.3e} "
                f"rel_err={sd['u3d_partition_error_local_rel_l2']:.3e} "
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
    rho_robin=1.0,
):
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

        border_triplets = set()

        # z-faces
        for ix in range(x_start, x_stop + 1):
            for iy in range(y_start, y_stop + 1):
                border_triplets.add((ix, iy, z_start))
                border_triplets.add((ix, iy, z_stop))

        # x-faces
        for iy in range(y_start, y_stop + 1):
            for iz in range(z_start, z_stop + 1):
                border_triplets.add((x_start, iy, iz))
                border_triplets.add((x_stop, iy, iz))

        # y-faces
        for ix in range(x_start, x_stop + 1):
            for iz in range(z_start, z_stop + 1):
                border_triplets.add((ix, y_start, iz))
                border_triplets.add((ix, y_stop, iz))

        p_rows = []
        p_cols = []
        p_data = []

        # Build P from structured border indices directly.
        for ix, iy, iz in border_triplets:
            assert (ix, iy, iz) in mesh_triplets, (
                f"Border index ({ix}, {iy}, {iz}) is missing from the original mesh."
            )

            for di in range(-1, 2):
                for dj in range(-1, 2):
                    for dk in range(-1, 2):
                        nix = ix + di
                        niy = iy + dj
                        niz = iz + dk
                        if (
                            0 <= nix < len(unique_x)
                            and 0 <= niy < len(unique_y)
                            and 0 <= niz < len(unique_z)
                        ):
                            assert (nix, niy, niz) in mesh_triplets, (
                                f"Neighbor index ({nix}, {niy}, {niz}) is missing "
                                "from the original mesh."
                            )

                            row_key = (ix, iy, iz)
                            col_key = (nix, niy, niz)
                            if row_key in triplet_to_global_dof and col_key in triplet_to_global_dof:
                                p_rows.append(triplet_to_global_dof[row_key])
                                p_cols.append(triplet_to_global_dof[col_key])
                                p_data.append(1.0)

        P = csr_matrix(
            (p_data, (p_rows, p_cols)),
            shape=(V.dim(), V.dim()),
        )
        if P.nnz:
            P.data[:] = 1.0
        subdomain["P"] = P

    subdomains = solve_partition_domain(
        subdomains, solver, rho_robin=rho_robin, print_summary=True
    )
    diagnose_exact_dirichlet(subdomains, solver, print_summary=True)

    # rho sweep: diagnose whether the Robin fixed point is consistent.
    # A trace-only condition converges to a rho-insensitive wrong limit; large
    # rho should degenerate toward Dirichlet and approach the diagnostic floor.
    hmax = solver.meshV.hmax()
    rho_base = float(solver.sigma3d) / max(float(hmax), 1e-30)
    print("")
    print("rho sweep (converged error vs Robin penalty):")
    print("  rho          | iters | avg rel local | rel global")
    for mult in (0.1, 1.0, 10.0, 100.0, 1000.0):
        rho = rho_base * mult
        hist = solve_schwarz_robin(
            subdomains, solver, rho_robin=rho, max_iter=60, tol=1e-8,
            print_summary=False,
        )
        it, delta, rl, rg = hist[-1]
        print(f"  {rho:.4e} | {it:5d} | {rl:.6e}  | {rg:.6e}")

    solve_schwarz_robin(subdomains, solver, print_summary=True)
    return subdomains





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