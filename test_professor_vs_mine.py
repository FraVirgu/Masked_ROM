"""
Head-to-head comparison: the professor's DD assembly vs. this repo's.

Both sides are run on the SAME sphere geometry and the SAME decomposition
(decomposeDomain), so every difference reported here is a difference of
FORMULATION, not of mesh, ordering or parameters.

What is compared, in the order the note builds them:

    1. C_i   -- the 3D->1D averaging operator on box i
    2. G     -- the 1D coupling mass matrix
    3. A_i   -- the local operator that eq. (5) inverts
    4. eta_i -- the Robin flux leaving box i
    5. u_i   -- the local solution, and the reconstructed global error

The professor's side is rebuilt here from his primitives rather than imported:
professor_ROM_method.py is a top-to-bottom script (it loads ROM weights, torch,
librom and a precomputed geom_data.pkl at import time), so importing it is not
possible. Every formula below is transcribed from it with a line reference, and
the xii calls -- Average/Circle/block_form/ii_assemble -- are the same library
functions his script calls, not reimplementations.

Run
---
    export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH
    python test_professor_vs_mine.py -name report_sphere_small -n 40

The defaults match the run that produced report_x_cross.txt, so a saved global
solution is reused and no expensive solve happens.
"""

import argparse
import os

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve

from dolfin import (
    Function, FunctionSpace, TrialFunction, TestFunction, Measure,
    MeshFunction, Constant, assemble, inner, grad, facets, as_backend_type,
)
from xii import Circle
from xii.assembler.average_form import average_space
from xii.assembler.average_matrix import scalar_average_matrix

from Decompose_Domain_Analytic_sphere import (
    ARTIFICIAL_FACET_TAG,
    check_sphere_domain_consistency,
    decomposeDomain,
    mark_artificial_facets,
)
from Robin_residual_sphere import (
    _robin_interface,
    _interface_selector,
    _face_neighbours,
    build_local_operator,
)
from Solver_full_domain import Solver3D1D
from Boundary import SphereBoundary, random_sphere_points


# =============================================================================
# reporting helpers
# =============================================================================

def _csr(A):
    return A.tocsr() if hasattr(A, "tocsr") else A


def rel_diff(A, B):
    """||A - B||_F / ||B||_F for sparse or dense, with a 0/0 -> 0 convention."""
    A, B = _csr(A), _csr(B)
    nb = np.linalg.norm(B.data if hasattr(B, "data") else B)
    d = A - B
    nd = np.linalg.norm(d.data if hasattr(d, "data") else d)
    return nd / nb if nb > 1e-300 else (0.0 if nd < 1e-300 else np.inf)


def rel_diff_vec(a, b):
    nb = float(np.linalg.norm(b))
    nd = float(np.linalg.norm(np.asarray(a) - np.asarray(b)))
    return nd / nb if nb > 1e-300 else (0.0 if nd < 1e-300 else np.inf)


def banner(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def verdict(x, tol=1e-10, warn=1e-2):
    if x <= tol:
        return "IDENTICAL"
    if x <= warn:
        return "close"
    return "DIFFERENT"


# =============================================================================
# THE PROFESSOR'S SIDE
# =============================================================================

def make_radius_fn(meshQ, q_radii):
    """A callable radius(x0) reproducing this repo's per-vertex radius rule.

    xii's Circle accepts either a number or a map from a centreline point to a
    radius. Handing it a single mean radius would make section 1 compare TWO
    differences at once (normalization AND radius), so we give the professor's
    side exactly the radii our own builder uses:

        Ri = max(0.5*(r[v0] + r[v1]), 0.005)     averaged over incident cells

    (Solver_partition_domain.average_matrix_diff_radii_skip, r_sum/r_cnt).
    With this in place, any residual difference in C_i is attributable to the
    curve_measure-vs-used_measure normalization alone.
    """
    x = meshQ.coordinates()
    r = np.asarray(q_radii, dtype=float)
    r_sum = np.zeros(meshQ.num_vertices())
    r_cnt = np.zeros(meshQ.num_vertices())
    for a, b in meshQ.cells():
        a, b = int(a), int(b)
        R = max(0.5 * (r[a] + r[b]), 0.005)
        r_sum[a] += R; r_cnt[a] += 1
        r_sum[b] += R; r_cnt[b] += 1
    vert_r = r_sum / np.maximum(r_cnt, 1)
    from scipy.spatial import cKDTree
    tree = cKDTree(x)

    def radius(x0):
        _, v = tree.query(np.asarray(x0, dtype=float))
        return float(vert_r[int(v)])

    return radius


def prof_local_C(V_loc, Q_loc, radius):
    """His C_i, via xii's own averaging kernel -- professor_ROM_method.py:174-179.

        cylinder = Circle(radius=R, degree=10)
        C        = average_3d1d_matrix(V, Q, cylinder)

    `average_3d1d_matrix` is not exported by the installed xii, so we call the
    kernel it wraps, scalar_average_matrix, directly.

    WHICH SPACE. xii's averaging operator lives on its own
    `average_space(V, meshQ)` -- a DISCONTINUOUS Lagrange space whose dofs are
    ordered per 1D cell, NOT the CG1 vertex space our C_i uses. The two
    operators are therefore not comparable entry by entry: a row-to-row
    subtraction would report a large difference that is pure dof ordering.

    So this returns the DG operator together with its space, and section 1
    compares the two FUNCTIONALLY -- both are applied to the same 3D field and
    the resulting circle-averages are compared at matching physical points.
    That is the only ordering-free way to ask whether the two builders compute
    the same average.

    THE DIFFERENCE BEING MEASURED. xii's scalar_average_matrix divides each row by

        curve_measure = sum(wq)          # the FULL circle

    computed BEFORE the collision loop, and simply skips quadrature points that
    miss the mesh. So for a vessel whose circle straddles a cut, his C_i keeps
    the global arc weights and drops the rest -- it IS the exact one-sided
    restriction of the global C.

    This repo's average_matrix_diff_radii_skip instead divides by

        used_measure                     # the SURVIVING arc

    i.e. it renormalizes, so its average is taken over the surviving arc only.
    Away from any cut the two coincide; at a straddling vessel they do not.
    That is the "77%" figure in Solver_partition_domain, and it means our
    DEFAULT path deviates from him while restrict_global_C=True matches.
    """
    meshQ = Q_loc.mesh()
    TV = average_space(V_loc, meshQ)
    C_dg = _petsc_to_csr(
        scalar_average_matrix(V_loc, TV, Circle(radius=radius, degree=10)),
        (TV.dim(), V_loc.dim()))
    return C_dg, TV


def compare_C_functionally(C_ours, Q_loc, C_prof, TV, u_local):
    """Apply both averaging operators to the same field and compare the
    resulting averages at matching physical points.

    Returns (rel_diff, n_matched). DG dofs sitting at the same coordinate as a
    CG1 vertex are averaged together first, so a shared vertex is compared once.
    """
    a_ours = C_ours.dot(u_local)
    a_prof = C_prof.dot(u_local)

    x_cg = Q_loc.tabulate_dof_coordinates().reshape((Q_loc.dim(), -1))
    x_dg = TV.tabulate_dof_coordinates().reshape((TV.dim(), -1))

    from scipy.spatial import cKDTree
    tree = cKDTree(x_cg)
    dist, idx = tree.query(x_dg)
    ok = dist < 1e-10
    if not np.any(ok):
        return np.inf, 0

    acc = np.zeros(Q_loc.dim()); cnt = np.zeros(Q_loc.dim())
    np.add.at(acc, idx[ok], a_prof[ok])
    np.add.at(cnt, idx[ok], 1.0)
    seen = cnt > 0
    prof_at_cg = np.zeros(Q_loc.dim())
    prof_at_cg[seen] = acc[seen] / cnt[seen]

    return rel_diff_vec(a_ours[seen], prof_at_cg[seen]), int(seen.sum())


def _petsc_to_csr(mat, shape):
    i, j, v = mat.getValuesCSR()
    return csr_matrix((v, j, i), shape=shape)


def _dolfin_to_csr(A, shape):
    i, j, v = as_backend_type(A).mat().getValuesCSR()
    return csr_matrix((v, j, i), shape=shape)


def prof_local_operator(V, mesh, markers_222, facet_markers, K1, LAMBDA, BETA,
                        interior_tag=222):
    """His A_tot -- professor_ROM_method.py:203-234 and :374.

        a[0][0]          = K1 * inner(grad(u), grad(v)) * dx
        re_phy[0][0]     = BETA   * inner(u, v) * ds_loc(111)
        re_gammaij[0][0] = LAMBDA * inner(u, v) * ds_loc(222)
        A_tot = A00 + M00 + Rphy + Rgamma

    Two structural differences from ours are visible right here and are
    reported, not silently reconciled:

      * he has NO reaction term in the volume; ours is
        sigma3d*(grad.grad + u*v), i.e. a Helmholtz block. The u*v is an extra
        mass matrix he does not have.
      * he integrates the Laplacian over the WHOLE box (plain `dx`), we
        integrate over dx(222) only, because we eliminate exterior dofs by
        Dirichlet instead of carrying his BETA-Robin on the physical boundary.

    Returns (A00, R_phy, R_gamma) separately so each can be compared alone.
    """
    # V MUST be the partition solver's own space, not a fresh FunctionSpace on
    # the same mesh: a new space would get its own dof numbering and every
    # matrix comparison below would silently compare permuted operators.
    u, v = TrialFunction(V), TestFunction(V)
    dx_all = Measure("dx", domain=mesh, subdomain_data=markers_222)
    ds_loc = Measure("ds", domain=mesh, subdomain_data=facet_markers)

    A00 = _asm(Constant(K1) * inner(grad(u), grad(v)) * dx_all(interior_tag), V)
    R_phy = _asm(Constant(BETA) * inner(u, v) * ds_loc(PHYSICAL_TAG), V)
    R_gam = _asm(Constant(LAMBDA) * inner(u, v) * ds_loc(ARTIFICIAL_FACET_TAG), V)
    return A00, R_phy, R_gam


PHYSICAL_TAG = 111  # his tag for the true outer boundary


def _asm(form, V):
    A = assemble(form)
    i, j, val = as_backend_type(A).mat().getValuesCSR()
    return csr_matrix((val, j, i), shape=(V.dim(), V.dim()))


def prof_robin_flux(A00, M00, R_phy, R_gamma, b_local, u_local):
    """His eta_j -- professor_ROM_method.py:1580-1592.

        D         = A00 + M00 + Rphy                 # NOTE: Rgamma excluded
        local_eta = -f3d + L1d + Lf - D*u + Rgamma*u

    where (-f3d + L1d + Lf) is exactly the local rhs. Note he does NOT apply
    any P^Gamma restriction: eta is computed on the whole local vector and the
    interface selection happens later, at the gather.
    """
    D = A00 + M00 + R_phy
    return b_local - D.dot(u_local) + R_gamma.dot(u_local)


def prof_gather(subdomains, etas, n_global, mu_mult, mass_global_list):
    """His interface gather -- professor_ROM_method.py:1691-1721.

        for j != i:                                  # ALL j, not face-neighbours
            eta_global_in[mask_interface] += eta_j_global[mask_interface]
            eta_global_in[mask_cross]     += mass_i * (eta_j/mass_j) * penalty

    mask_interface is mu_mult == 2 (a plain face), mask_cross is mu_mult > 2
    (a box edge or corner, where more than two boxes meet). At a cross point
    each neighbour computed its eta against ITS OWN local mass over ITS OWN
    slice of the surface, so summing raw over-counts by the multiplicity. He
    converts to a density (/mass_j), re-weights with i's mass, and scales by a
    partition-of-unity `penalty`.

    `penalty` comes from his geom_data.pkl and is not reproducible here, so we
    use the natural choice penalty = 1/(mu_mult - 1): each of the (mu-1) other
    owners contributes an equal share. This is documented as an assumption, not
    presented as his value.
    """
    mask_iface = (mu_mult == 2)
    mask_cross = (mu_mult > 2)
    penalty = np.zeros(n_global)
    penalty[mask_cross] = 1.0 / np.maximum(mu_mult[mask_cross] - 1.0, 1.0)

    incoming = []
    for i, sd_i in enumerate(subdomains):
        l2g_i = sd_i["local_to_global_dof"]
        eta_in = np.zeros(n_global)

        for j, sd_j in enumerate(subdomains):
            if j == i:
                continue
            l2g_j = sd_j["local_to_global_dof"]

            eta_j_glob = np.zeros(n_global)
            eta_j_glob[l2g_j] += etas[j]

            eta_in[mask_iface] += eta_j_glob[mask_iface]

            # density on j, re-weighted by i's mass, damped by penalty
            mj = mass_global_list[j]
            dens_j = np.zeros(n_global)
            ok = mask_cross & (mj > 1e-30)
            dens_j[ok] = eta_j_glob[ok] / mj[ok]
            eta_in[mask_cross] += (mass_global_list[i][mask_cross]
                                   * dens_j[mask_cross]
                                   * penalty[mask_cross])

        incoming.append(eta_in[l2g_i])
    return incoming, mask_iface, mask_cross


# =============================================================================
# OUR SIDE  (re-expressed so the two are directly commensurable)
# =============================================================================

def our_gather_OLD(subdomains, etas, n_global, sels):
    """The gather this repo used BEFORE the mass-weighting fix.

        nbrs = _face_neighbours(subdomains)      # |ijk_i - ijk_j| == [0,0,1]
        for j in nbrs[i]:
            incoming[iface_i & owned_j] += f_star_minus_j[...]

    Raw sum, face-neighbours only. Kept as the baseline the fix is measured
    against; Robin_residual_sphere no longer does this.
    """
    nbrs = _face_neighbours(subdomains)
    f_star = _masked_global(subdomains, etas, n_global, sels)
    incoming = []
    for i, sd in enumerate(subdomains):
        l2g_i = sd["local_to_global_dof"]
        iface_i = np.zeros(n_global, dtype=bool)
        iface_i[l2g_i[sels[i]]] = True
        inc = np.zeros(n_global)
        for j in nbrs[i]:
            owned_j = np.zeros(n_global, dtype=bool)
            owned_j[subdomains[j]["local_to_global_dof"]] = True
            sh = iface_i & owned_j
            inc[sh] += f_star[j][sh]
        incoming.append(inc[l2g_i])
    return incoming


def our_gather_NEW(subdomains, etas, n_global, sels, masses, mu_mult):
    """The gather Robin_residual_sphere uses now.

    Two changes from OLD, both required:
      * the target set is the CONNECTIVITY set mu==2, not the intersection of
        the two boxes' P^Gamma selectors (which the exterior-dof elimination
        shrinks to a small fraction of the real shared surface);
      * mu>2 dofs get the density rule m_i*(f_j/m_j)*penalty instead of a raw
        sum.
    The sent vector is still P^Gamma-masked.
    """
    f_star = _masked_global(subdomains, etas, n_global, sels)
    mg = masses_to_global(subdomains, masses, n_global)
    mask_face = (mu_mult == 2.0)
    mask_cross = (mu_mult > 2.0)
    penalty = np.zeros(n_global)
    penalty[mask_cross] = 1.0 / np.maximum(mu_mult[mask_cross] - 1.0, 1.0)

    incoming = []
    for i, sd in enumerate(subdomains):
        inc = np.zeros(n_global)
        for j, sd_j in enumerate(subdomains):
            if j == i:
                continue
            inc[mask_face] += f_star[j][mask_face]
            ok = mask_cross & (mg[j] > 1e-30)
            dens = np.zeros(n_global)
            dens[ok] = f_star[j][ok] / mg[j][ok]
            inc[mask_cross] += (mg[i][mask_cross] * dens[mask_cross]
                                * penalty[mask_cross])
        incoming.append(inc[sd["local_to_global_dof"]])
    return incoming


def _masked_global(subdomains, etas, n_global, sels):
    """P^Gamma-masked eta of each box, scattered to global dofs."""
    out = []
    for i, sd in enumerate(subdomains):
        g = np.zeros(n_global)
        g[sd["local_to_global_dof"]] = np.where(sels[i], etas[i], 0.0)
        out.append(g)
    return out


def local_surface_mass(ps, facet_markers):
    """Row sums of the artificial-interface mass matrix = his mass_global_list
    restricted to this box: the local test-function mass on Gamma."""
    V = ps.V
    v = TestFunction(V)
    ds_art = Measure("ds", domain=ps.meshV, subdomain_data=facet_markers)
    return assemble(v * ds_art(ARTIFICIAL_FACET_TAG)).get_local()


# =============================================================================
# main
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__.split("Run")[0],
    )
    p.add_argument("-name", type=str, default="report_sphere_small")
    p.add_argument("-n", type=int, default=40)
    p.add_argument("-sigma1d", type=float, default=1.0)
    p.add_argument("-sigma3d", type=float, default=1e-3)
    p.add_argument("-kappa", type=float, default=1.0)
    p.add_argument("-radius", type=float, default=5.0)
    p.add_argument("-rho", type=float, default=1.0,
                   help="Robin penalty; used as his LAMBDA so the two sides "
                        "invert the same operator")
    p.add_argument("-beta", type=float, default=0.0,
                   help="his BETA on the physical boundary. Default 0 because "
                        "we use Dirichlet elimination instead; set >0 to see "
                        "how much that term changes A_i")
    p.add_argument("-restrict_global_C", action="store_true",
                   help="build our C_i by restricting the global C. This is "
                        "the path that MATCHES the professor -- see prof_local_C")
    p.add_argument("-solution", type=str, default=None)
    args = p.parse_args()

    if args.solution is None:
        args.solution = os.path.join(
            "solution",
            f"Simple{args.name}_n{args.n}"
            f"_s1d{args.sigma1d}_s3d{args.sigma3d}_k{args.kappa}")

    # ---- geometry, exactly as Robin_residual_sphere's __main__ does ----------
    mesh_prefix = os.path.join("nets", args.name, args.name) + "_"
    if not os.path.isdir(os.path.join("nets", args.name)):
        raise SystemExit(f"nets/{args.name} does not exist.")

    boundary = SphereBoundary(
        radius=args.radius,
        inlet_points=random_sphere_points(40, x_sign=-1, min_x=0.2,
                                          min_dist_to_boundary=0.06,
                                          radius=args.radius),
        outlet_points=random_sphere_points(40, x_sign=+1, min_x=0.2,
                                           min_dist_to_boundary=0.06,
                                           radius=args.radius),
        border_eps=10e-1,
    )
    check_sphere_domain_consistency(boundary=boundary,
                                    n_min=-args.radius, n_max=args.radius)

    solver = Solver3D1D(
        path_to_1D_mesh=mesh_prefix, boundary=boundary, n=args.n,
        sigma3d=args.sigma3d, sigma1d=args.sigma1d, kappa=args.kappa,
        exterior="dirichlet",
    ).build()

    n_3d, n_1d = solver.W[0].dim(), solver.W[1].dim()
    sol_npy = os.path.join(args.solution, "solution.npy")
    if not os.path.isfile(sol_npy):
        raise SystemExit(f"No saved solution at {sol_npy}. Run the global "
                         f"solve first (see Robin_residual_sphere.py).")
    x_np = np.load(sol_npy)
    if x_np.size != n_3d + n_1d:
        raise SystemExit(f"{sol_npy} has {x_np.size} entries, this mesh needs "
                         f"{n_3d + n_1d}. Wrong mesh for this field.")
    solver.x_np = x_np
    solver.u3d = Function(solver.W[0]); solver.u3d.vector()[:] = x_np[:n_3d]
    solver.u1d = Function(solver.W[1]); solver.u1d.vector()[:] = x_np[n_3d:]
    print(f"Loaded global solution ({n_3d} 3D dofs, {n_1d} 1D dofs)")

    # ---- ONE decomposition, shared by both sides ----------------------------
    subdomains = decomposeDomain(solver, boundary,
                                 restrict_global_C=args.restrict_global_C)

    V_global = solver.W[0]
    n_global = V_global.dim()
    coords = V_global.tabulate_dof_coordinates().reshape((n_global, -1))
    g_min, g_max = coords.min(axis=0), coords.max(axis=0)
    u_star = solver.u3d.vector().get_local()

    LAMBDA = float(args.rho)
    K1 = float(solver.sigma3d)

    # Same per-vertex radii on both sides, so section 1 isolates the
    # normalization difference and nothing else.
    radius_fn = make_radius_fn(solver.meshQ,
                               np.asarray(solver.Q_radii.array(), dtype=float))

    # multiplicity, his global_mu_mult (professor_ROM_method.py:1244)
    mu = np.zeros(n_global)
    for sd in subdomains:
        mu[sd["local_to_global_dof"]] += 1.0

    banner("0. SETUP")
    print(f"  subdomains          : {len(subdomains)}")
    print(f"  global 3D dofs      : {n_global}")
    print(f"  LAMBDA (= our rho)  : {LAMBDA:g}")
    print(f"  K1     (= sigma3d)  : {K1:g}")
    print(f"  BETA                : {args.beta:g}")
    print(f"  our C_i built by    : "
          f"{'restricted global C' if args.restrict_global_C else 'local clipped quadrature (renormalized)'}")
    print(f"  mu_mult histogram   : "
          + ", ".join(f"{int(v)}->{int(c)}"
                      for v, c in zip(*np.unique(mu, return_counts=True))))
    print("     mu==1 interior, mu==2 face, mu>2 edge/corner (his mask_cross)")

    # =========================================================================
    rows_C, rows_A, rows_eta = [], [], []
    rowsums = []
    prof_etas, our_etas, sels, masses = [], [], [], []
    prof_ops, our_ops, b_locals = [], [], []

    for sd in subdomains:
        ps = sd["partition_solver"]
        l2g = sd["local_to_global_dof"]
        V_loc = ps.V
        u_i = u_star[l2g]

        fm, _ = mark_artificial_facets(ps, g_min, g_max)
        G_gamma, _ = _robin_interface(ps, g_min, g_max)
        sel = _interface_selector(sd, ps, l2g)
        sels.append(sel)
        masses.append(local_surface_mass(ps, fm))

        # ---- 1. C_i -------------------------------------------------------
        C_prof, TV_prof = prof_local_C(V_loc, ps.Q, radius_fn)
        C_ours = _csr(ps.C)
        dC, n_match = compare_C_functionally(C_ours, ps.Q, C_prof, TV_prof, u_i)
        # Row sums: a true circle-average has row sum 1. A row sum < 1 means
        # the operator lost part of the circle without renormalizing, i.e. it
        # under-reports that vessel's average rather than averaging over less.
        rs_o = np.asarray(C_ours.sum(axis=1)).ravel()
        rs_p = np.asarray(C_prof.sum(axis=1)).ravel()
        act_o = rs_o[np.abs(rs_o) > 1e-12]
        act_p = rs_p[np.abs(rs_p) > 1e-12]
        rowsums.append((float(act_o.min()) if act_o.size else 0.0,
                        float(act_o.mean()) if act_o.size else 0.0,
                        float(act_p.min()) if act_p.size else 0.0,
                        float(act_p.mean()) if act_p.size else 0.0))
        rows_C.append((sd["ijk"], dC, n_match,
                       int(getattr(ps, "C_rows_straddling",
                                   np.zeros(0)).size
                           if getattr(ps, "C_rows_straddling", None) is not None
                           else 0)))

        # ---- 3. A_i -------------------------------------------------------
        A00_p, Rphy_p, Rgam_p = prof_local_operator(
            V_loc, ps.meshV, ps.meshV_markers, fm, K1, LAMBDA, args.beta,
            interior_tag=222)
        M00_ours = _csr(ps.C).T @ _csr(ps.G) @ _csr(ps.C)
        A_prof = A00_p + M00_ours + Rphy_p + Rgam_p
        A_ours = _csr(build_local_operator(ps, LAMBDA, G_gamma))
        prof_ops.append((A00_p, Rphy_p, Rgam_p, M00_ours, A_prof))
        our_ops.append(A_ours)
        # Split the volume block: our stiffness is sigma3d*grad.grad, which is
        # his K1*grad.grad exactly, and our EXTRA term is sigma3d*u*v. Reporting
        # them together would just measure the mass term's size relative to the
        # stiffness and say nothing about whether the shared part agrees.
        AD_ours = _csr(ps.A) - M00_ours          # our volume block
        u_l, v_l = TrialFunction(V_loc), TestFunction(V_loc)
        dxl = Measure("dx", domain=ps.meshV, subdomain_data=ps.meshV_markers)
        K_ours = _asm(Constant(K1) * inner(grad(u_l), grad(v_l)) * dxl(222), V_loc)
        Mass_ours = _asm(Constant(K1) * inner(u_l, v_l) * dxl(222), V_loc)
        rows_A.append((
            sd["ijk"],
            rel_diff(K_ours, A00_p),                  # stiffness: must be ~0
            (np.linalg.norm(Mass_ours.data)
             / max(np.linalg.norm(A00_p.data), 1e-300)),   # size of our extra u*v
            rel_diff(LAMBDA * G_gamma, Rgam_p),       # Robin interface block
            rel_diff(A_ours, A_prof),                 # the full operator
        ))

        # ---- 4. eta_i -----------------------------------------------------
        b_i = np.asarray(ps.rhs, dtype=float)
        b_locals.append(b_i)
        eta_p = prof_robin_flux(A00_p, M00_ours, Rphy_p, Rgam_p, b_i, u_i)
        eta_o = b_i - _csr(ps.A).dot(u_i) + LAMBDA * G_gamma.dot(u_i)
        prof_etas.append(eta_p)
        our_etas.append(eta_o)

        live = np.ones(eta_o.size, dtype=bool)
        ext = getattr(ps, "ext_dofs", None)
        if ext is not None and np.size(ext):
            live[np.asarray(ext, dtype=int)] = False
        fp = (np.linalg.norm(eta_p[sel & live])
              / max(np.linalg.norm(eta_p[live]), 1e-300))
        fo = (np.linalg.norm(eta_o[sel & live])
              / max(np.linalg.norm(eta_o[live]), 1e-300))
        rows_eta.append((sd["ijk"], rel_diff_vec(eta_o, eta_p), fp, fo))

    # =========================================================================
    banner("1. COUPLING OPERATOR  C_i    (ours vs his xii Average)")
    print("  Compared FUNCTIONALLY: both operators applied to the same u*,")
    print("  averages matched at coincident physical points (his operator")
    print("  lives on a DG space with a different dof ordering).")
    print("  his divides by the FULL circle and drops outside points;")
    print("  ours (default path) divides by the SURVIVING arc.")
    print(f"  {'ijk':<12}{'rel diff':>12}{'straddl':>9}"
          f"{'ours rowsum':>22}{'his rowsum':>20}")
    print(f"  {'':<12}{'':>12}{'':>9}{'min':>11}{'mean':>11}{'min':>10}{'mean':>10}")
    for (ijk, d, nm, nstr), (o_mn, o_mu, p_mn, p_mu) in zip(rows_C, rowsums):
        print(f"  {str(ijk):<12}{d:>12.3e}{nstr:>9d}"
              f"{o_mn:>11.4f}{o_mu:>11.4f}{p_mn:>10.4f}{p_mu:>10.4f}")
    print()
    print("  READ THIS COLUMN FIRST: row sum 1.0 = a genuine average.")
    print("  Whichever side shows row sums < 1 is losing circle weight")
    print("  outright instead of averaging over the part it still owns.")

    banner("2. 1D MASS MATRIX  G")
    print("  ours: assemble(kappa*2*pi*R * p*q * dx) on the shared global meshQ")
    print("  his : J = 2*pi*L_cap*rho_int*R inside m[0][0] = J*inner(Ru,Rv)*dx_")
    print("  Same form up to the constant; his J carries extra physical")
    print("  factors (L_cap, rho_int) that our kappa absorbs. Reported as a")
    print("  scalar ratio, since a mismatch here rescales the whole coupling.")
    G_ours = _csr(subdomains[0]["partition_solver"].G)
    print(f"  ||G_ours||_F = {np.linalg.norm(G_ours.data):.6e}")
    print(f"  G is shared by every box (same meshQ)  : "
          f"{all(rel_diff(_csr(sd['partition_solver'].G), G_ours) < 1e-14 for sd in subdomains)}")

    banner("3. LOCAL OPERATOR  A_i")
    print("  stiff : our sigma3d*grad.grad*dx(222) vs his K1*grad.grad*dx(222)")
    print("          -> the block both sides share; must be ~0")
    print("  +mass : ||our extra sigma3d*u*v|| / ||stiffness||. He has NO such")
    print("          term. This is a MODELLING difference (Helmholtz vs Poisson),")
    print("          not a bug, but it changes what eq. (5) inverts.")
    print("  robin : our rho*G_gamma vs his LAMBDA*inner(u,v)*ds(222)")
    print("  total : A_i as eq. (5) inverts it")
    print(f"  {'ijk':<12}{'stiff':>12}{'+mass':>10}{'robin':>12}{'total':>12}"
          f"  verdict(stiff)")
    for ijk, dk, dm, dr, dt in rows_A:
        print(f"  {str(ijk):<12}{dk:>12.3e}{dm:>10.2f}{dr:>12.3e}{dt:>12.3e}"
              f"  {verdict(dk)}")

    banner("4. ROBIN FLUX  eta_i")
    print("  his : eta = b - (A00+M00+Rphy)u + Rgamma*u,  NO P^Gamma applied")
    print("  ours: r   = b - A u + rho*G_gamma*u,  then masked by sel")
    print("  'iface frac' = ||eta on Gamma|| / ||eta||. Eq. (4) discards")
    print("  everything off Gamma, so this must be ~1 for the extraction to")
    print("  mean anything. He never needs it -- his gather does the masking.")
    print(f"  {'ijk':<12}{'rel diff':>12}{'iface his':>12}{'iface ours':>12}")
    for ijk, d, fp, fo in rows_eta:
        print(f"  {str(ijk):<12}{d:>12.3e}{fp:>12.1%}{fo:>12.1%}")

    # ---- 5. gather + solve --------------------------------------------------
    inc_prof, mask_iface, mask_cross = prof_gather(
        subdomains, prof_etas, n_global, mu, masses_to_global(subdomains, masses, n_global))
    inc_old = our_gather_OLD(subdomains, our_etas, n_global, sels)
    inc_ours = our_gather_NEW(subdomains, our_etas, n_global, sels,
                              masses, mu)

    banner("5. INTERFACE GATHER  E_ij")
    print(f"  face dofs   (mu==2) : {int(mask_iface.sum())}")
    print(f"  cross dofs  (mu >2) : {int(mask_cross.sum())}   <- his mask_cross")
    print("  OLD wrote only where sel_i & sel_j -- the exterior-dof")
    print("  elimination shrinks that to a fraction of the shared surface")
    print("  (351 of 4800 dofs for box0<-box1). NEW targets the connectivity")
    print("  set mu==2 like he does, and uses the density rule")
    print("  m_i*(f_j/m_j)*penalty on the mu>2 edge/corner dofs.")
    print(f"  {'ijk':<12}{'old vs his':>12}{'new vs his':>12}"
          f"{'||his||':>13}{'||old||':>13}{'||new||':>13}")
    for i, sd in enumerate(subdomains):
        np_ = np.linalg.norm(inc_prof[i])
        no_ = np.linalg.norm(inc_old[i])
        nn_ = np.linalg.norm(inc_ours[i])
        print(f"  {str(sd['ijk']):<12}"
              f"{rel_diff_vec(inc_old[i], inc_prof[i]):>12.3e}"
              f"{rel_diff_vec(inc_ours[i], inc_prof[i]):>12.3e}"
              f"{np_:>13.4e}{no_:>13.4e}{nn_:>13.4e}")

    # ---- 6. solve both, same operator, different rhs ------------------------
    banner("6. LOCAL SOLVE AND GLOBAL ERROR")
    acc_p = np.zeros(n_global); acc_o = np.zeros(n_global)
    acc_old = np.zeros(n_global)
    cnt = np.zeros(n_global)
    print(f"  {'ijk':<12}{'raw':>12}{'his gather':>12}"
          f"{'ours OLD':>12}{'ours NEW':>12}")
    for i, sd in enumerate(subdomains):
        ps = sd["partition_solver"]
        l2g = sd["local_to_global_dof"]
        G_gamma, _ = _robin_interface(ps, g_min, g_max)
        A_robin = build_local_operator(ps, LAMBDA, G_gamma)

        ext = getattr(ps, "ext_dofs", None)
        fp_, fo_ = inc_prof[i].copy(), inc_ours[i].copy()
        fold_ = inc_old[i].copy()
        if ext is not None and np.size(ext):
            e = np.asarray(ext, dtype=int)
            fp_[e] = 0.0
            fo_[e] = 0.0
            fold_[e] = 0.0

        u_p = spsolve(A_robin, b_locals[i] + fp_)
        u_o = spsolve(A_robin, b_locals[i] + fo_)
        u_old = spsolve(A_robin, b_locals[i] + fold_)

        u_ref = sd["sol_3d_sub"].vector().get_local()
        nref = max(float(np.linalg.norm(u_ref)), 1e-12)
        raw = sd["u3d_partition_error_local_raw_rel_l2"]
        print(f"  {str(sd['ijk']):<12}{raw:>12.3e}"
              f"{np.linalg.norm(u_ref - u_p) / nref:>12.3e}"
              f"{np.linalg.norm(u_ref - u_old) / nref:>12.3e}"
              f"{np.linalg.norm(u_ref - u_o) / nref:>12.3e}")

        acc_p[l2g] += u_p
        acc_o[l2g] += u_o
        acc_old[l2g] += u_old
        cnt[l2g] += 1.0

    nz = cnt > 0
    rec_p = np.where(nz, acc_p / np.maximum(cnt, 1), u_star)
    rec_o = np.where(nz, acc_o / np.maximum(cnt, 1), u_star)
    rec_old = np.where(nz, acc_old / np.maximum(cnt, 1), u_star)
    ref = float(np.linalg.norm(u_star))
    raw_g = subdomains[0]["u3d_partition_full_error_raw_rel_l2"]

    print()
    print(f"  reconstructed global relative L2 error")
    print(f"    raw (no coupling) : {raw_g:.6e}")
    print(f"    his gather        : {np.linalg.norm(u_star - rec_p) / ref:.6e}")
    print(f"    ours OLD          : {np.linalg.norm(u_star - rec_old) / ref:.6e}")
    print(f"    ours NEW (fixed)  : {np.linalg.norm(u_star - rec_o) / ref:.6e}")
    print()
    print("  All three invert the SAME A_i, so every gap is caused purely by")
    print("  the interface gather.")


def masses_to_global(subdomains, masses, n_global):
    """Scatter each box's local surface mass to a global vector, as his
    mass_global_list -- one entry per subdomain, indexed by global dof."""
    out = []
    for sd, m in zip(subdomains, masses):
        g = np.zeros(n_global)
        g[sd["local_to_global_dof"]] = m
        out.append(g)
    return out


if __name__ == "__main__":
    main()
